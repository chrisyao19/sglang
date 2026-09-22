// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lifecycle bundle for the KV-event index.
//!
//! Couples the three submodules that are independent in their own right but
//! always operate together in production:
//!
//! - [`HashTree`] — the cache-aware routing index keyed by SGLang block hash.
//! - [`EngineReportedLoadTable`] — engine-reported per-worker load.
//! - Two [`KvEventSubscriberRegistry`]s — one per `(worker_url, dp_rank)` on
//!   the cache topic, one on the load topic.
//! - A pump task that drains [`WorkerEvent`]s and applies KV batches to the
//!   tree and `Load` snapshots to the engine-load table.
//!
//! `add_worker` / `remove_worker` are driven from the worker manager on every
//! `DiscoveryEvent::Added` / `DiscoveryEvent::Removed`.
//!
//! # Race avoidance
//!
//! The pump runs independently of the lifecycle calls, so an event can sit in
//! the mpsc buffer while `remove_worker` is in progress. To prevent stale
//! events from re-inserting tree state for a worker that was just torn down,
//! [`KvEventIndex`] maintains a `live_workers` set; entries are removed
//! **before** the subscriber tasks are joined, and the pump filters every
//! event through this set before mutating the tree.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use flate2::write::GzEncoder;
use flate2::Compression;
use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::block_size_oracle::BlockSizeOracle;
use super::bootstrap::{
    BootstrapState, BootstrapTracker, PeerRegistry, PeerSnapshot, RankOutcome, VettedSnapshot,
    WireWorker, SNAPSHOT_FORMAT,
};
use super::discovery::{fetch_event_config, EventConfig};
use super::subscriber::{KvEventSubscriberRegistry, SubKind, WorkerEvent};
use super::tally::{EventKind, EventTally};
use super::tree::{HashTree, KvWorkerId, SnapshotNode, Tiers};
use super::wire::{KvCacheEvent, KvEventBatch};
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;

/// Channel buffer between the subscriber registry and the pump task.
///
/// Bounded so a misbehaving publisher cannot exhaust memory.  Realistic
/// per-worker event rates are < 1 kHz; a 1024-deep buffer absorbs a
/// half-second burst at 2 kHz before back-pressuring the SUB sockets.
const EVENT_CHANNEL_BUFFER: usize = 1024;

/// Per-rank cap on batches held back while that rank is
/// [`BootstrapState::Pending`].
///
/// A rank that overflows this cannot be spliced (see `fail_rank`), so the
/// cap trades a small amount of memory for the chance to bootstrap at all.
/// Sized to match `EVENT_CHANNEL_BUFFER`: if the pump is that far behind, the
/// snapshot is not arriving in time anyway.
const PENDING_BATCH_LIMIT: usize = 1024;

// Constructed by the peer sweep, which lands next.
#[allow(dead_code)]
/// Control-plane messages for the pump task.
///
/// Tree mutation MUST stay on the single writer (see the single-writer property
/// in [`super::tree`]), so the bootstrap task never touches the tree itself —
/// it fetches and vets a snapshot, then hands it to the pump through this
/// channel.
#[derive(Debug)]
enum PumpControl {
    /// Graft a vetted snapshot, seed cursors, then release each rank's held
    /// batches.
    ///
    /// `obligations` is the set this message discharges: every rank named here
    /// leaves `Pending` when this is handled, whether or not the snapshot covered
    /// it. Deriving the set from the snapshot instead would leave a rank the peer
    /// never mentioned buffering forever.
    ///
    /// Each entry carries the incarnation it was registered under, so a task
    /// still in flight for a worker that has since been removed and re-added
    /// cannot graft onto the new incarnation.
    ApplySnapshot {
        obligations: Vec<(KvWorkerId, u64)>,
        vetted: Box<VettedSnapshot>,
    },
    /// Stop holding batches for `ranks`: release what is buffered and mark
    /// them [`BootstrapState::Failed`]. Sent when no peer could supply a
    /// snapshot, or when the bootstrap deadline fires.
    AbandonBootstrap { obligations: Vec<(KvWorkerId, u64)> },
    /// A worker was removed: drop everything the pump keeps for these ranks —
    /// held batches, pending splice proofs, tree carriers and cursors — so a
    /// re-added worker's fresh publisher inherits none of it.
    ///
    /// `done` fires once teardown has run, so nothing of these ranks survives
    /// `remove_worker`.
    ForgetRanks {
        ranks: Vec<KvWorkerId>,
        done: Option<oneshot::Sender<()>>,
    },
}

/// Per-worker bookkeeping kept inside [`KvEventIndex`] so `remove_worker`
/// knows which DP ranks were actually subscribed (not the advertised
/// `dp_size`, which may overflow `u16` and skip ranks).
#[derive(Debug, Clone)]
struct WorkerEntry {
    /// DP ranks that were successfully spawned for this worker. Used by
    /// `remove_worker` to know which `(url, dp_rank)` cursors and tree
    /// states to clear.
    dp_ranks: Vec<u32>,
}

/// Ranks whose socket port is representable for a publisher range. This is
/// shared by lifecycle bookkeeping and the subscriber registry contract so an
/// expected load rank always has a corresponding SUB socket.
fn subscribable_ranks(port_base: u16, dp_size: u32) -> Vec<u32> {
    let port_base = u32::from(port_base);
    (0..dp_size)
        .filter(|rank| port_base.saturating_add(*rank) <= u32::from(u16::MAX))
        .collect()
}

/// The read-only handles the `/metrics` scrape pulls the KV storage-tier
/// series from. Narrower than an [`KvEventIndex`] handle on purpose: a route
/// has no business calling `add_worker` / `remove_worker` / `shutdown`.
#[derive(Clone)]
pub struct KvIndexMetrics {
    pub(crate) tree: Arc<HashTree>,
    pub(crate) tally: Arc<EventTally>,
}

impl KvIndexMetrics {
    pub fn new(tree: Arc<HashTree>, tally: Arc<EventTally>) -> Self {
        Self { tree, tally }
    }
}

/// A built snapshot with both of its wire encodings.
///
/// The encodings are cached, not the struct: serialising and compressing a
/// fleet-sized tree costs about as much as the walk that produced it, so a
/// boot herd served from this cache pays each once per build, not per request.
struct CachedSnapshot {
    /// Taken before the cursors are read, so it never overstates what the
    /// snapshot covers.
    exported_at: Instant,
    body: SnapshotBody,
}

/// One snapshot encoded for the wire: the JSON document and its gzip.
///
/// Both empty means the encode failed; see [`encode_snapshot`].
#[derive(Debug, Clone, Default)]
pub struct SnapshotBody {
    /// The JSON document, served when the caller does not accept gzip.
    pub identity: Bytes,
    /// `identity`, gzip-compressed.
    pub gzip: Bytes,
}

impl SnapshotBody {
    /// Whether this is the encode-failure sentinel.
    pub fn is_empty(&self) -> bool {
        self.identity.is_empty()
    }
}

/// Whether an export sampled at `exported_at` meets a `max_age` a caller
/// stated when it arrived at `requested_at`.
///
/// Measured from arrival, not from whenever the caller gets the cache lock:
/// its requirement is a fixed point in time (when its ranks began holding),
/// so an export sampled after it arrived always meets it, however long a build
/// kept it queued.
fn meets_max_age(exported_at: Instant, requested_at: Instant, max_age: Duration) -> bool {
    exported_at >= requested_at || requested_at.duration_since(exported_at) < max_age
}

/// Serialise a snapshot to JSON.
///
/// `PeerSnapshot` is a plain `Serialize` struct with no non-string map keys,
/// so this cannot actually fail; an empty body on the impossible branch keeps
/// the caller total, and the route turns it into a non-success status rather
/// than a 200 a consumer would fail to decode.
fn encode_json(snap: &PeerSnapshot) -> Bytes {
    match serde_json::to_vec(snap) {
        Ok(v) => Bytes::from(v),
        Err(e) => {
            warn!(error = %e, "kv-bootstrap: snapshot serialisation failed");
            Bytes::new()
        }
    }
}

/// Encode a snapshot as JSON and as its gzip, or the empty sentinel if either
/// step fails.
///
/// Gzip runs at the fastest level: the compressibility is in the per-node JSON
/// scaffolding, so the lowest level already shrinks the body several-fold, and
/// this runs on a replica that is also serving traffic.
fn encode_snapshot(snap: &PeerSnapshot) -> SnapshotBody {
    let identity = encode_json(snap);
    if identity.is_empty() {
        return SnapshotBody::default();
    }
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    match encoder.write_all(&identity).and_then(|()| encoder.finish()) {
        Ok(gzip) => SnapshotBody {
            identity,
            gzip: Bytes::from(gzip),
        },
        Err(e) => {
            warn!(error = %e, "kv-bootstrap: snapshot compression failed");
            SnapshotBody::default()
        }
    }
}

/// Bundle of `HashTree` + `KvEventSubscriberRegistry` + pump task.
///
/// Construct one instance per router process and hand it to the worker
/// manager as `Option<Arc<KvEventIndex>>` — `None` disables the cache-aware
/// routing path entirely.
pub struct KvEventIndex {
    tree: Arc<HashTree>,
    maintain_tree: bool,
    subscribers: Arc<KvEventSubscriberRegistry>,
    /// Second registry subscribing to the load topic (one per worker rank),
    /// feeding `LoadStat` snapshots into `engine_reported_load`. Shares the pump
    /// channel with `subscribers`; keyed independently so KV and load
    /// subscribers for the same worker don't collide.
    load_subscribers: Arc<KvEventSubscriberRegistry>,
    /// Engine-reported per-worker load, written by the pump from
    /// `WorkerEvent::Load` and captured at request ingress.
    engine_reported_load: Arc<EngineReportedLoadTable>,
    pump: Mutex<Option<JoinHandle<()>>>,
    pump_cancel: CancellationToken,
    workers: Mutex<HashMap<String, WorkerEntry>>,
    http: reqwest::Client,
    /// Set of currently-attached `(worker_url, dp_rank)` pairs. The pump
    /// drops any event whose `worker` is not in this set, so a batch
    /// queued by a subscriber that was torn down by `remove_worker` does
    /// not re-pollute the tree after `clear_worker` ran.
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    /// Per-`(worker_url, dp_rank)` last-applied sequence number. The
    /// subscriber forwards every batch with no de-dup; this map filters
    /// any batch whose `seq` is not strictly greater than the previously
    /// applied one. Cleared on `remove_worker` because a re-added worker
    /// may legitimately have a fresh publisher whose sequence numbers
    /// restart from 1.
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    /// Applied events by kind and storage medium, for the `/metrics` scrape.
    /// Written only by the pump.
    tally: Arc<EventTally>,
    /// Per-rank bootstrap progress; also what `/readyz` consults.
    bootstrap: Arc<BootstrapTracker>,
    /// Sibling replicas a snapshot may be pulled from. See [`PeerRegistry`].
    peers: Arc<PeerRegistry>,
    /// Control channel into the pump, so snapshot grafting happens on the
    /// single writer rather than in the bootstrap task.
    ctrl_tx: mpsc::Sender<PumpControl>,
    /// Last built snapshot; see [`KvEventIndex::peer_snapshot_body`].
    snapshot_cache: Arc<AsyncMutex<Option<CachedSnapshot>>>,
    /// Worker-sourced `page_size` shared with prefix providers.
    /// `add_worker` calls `try_set(cfg.block_size)` so the first worker
    /// establishes the value; subsequent workers that disagree are
    /// rejected (logged + not subscribed). Prefix providers read it at routing
    /// time to size their `compute_block_hashes` calls.
    block_size_oracle: Arc<BlockSizeOracle>,
}

impl KvEventIndex {
    /// Build an empty index and spawn the pump task.
    pub fn new() -> Arc<Self> {
        Self::new_with_http(
            reqwest::Client::builder()
                .timeout(Duration::from_secs(2))
                .build()
                .expect("default http client builds"),
        )
    }

    /// Constructor used by tests so they can supply a custom timeout.
    pub fn new_with_http(http: reqwest::Client) -> Arc<Self> {
        Self::new_with_http_and_oracle(http, BlockSizeOracle::new())
    }

    /// Constructor that lets the caller supply a pre-shared
    /// [`BlockSizeOracle`]. Production wires this from `AppContext` so
    /// the same oracle the index seeds is available to prefix providers.
    /// Tests use this to pre-populate the
    /// oracle and exercise the mismatch-rejection path.
    pub fn new_with_http_and_oracle(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Arc<Self> {
        Self::new_with_mode(
            http,
            block_size_oracle,
            Arc::new(BootstrapTracker::disabled()),
            true,
        )
    }

    /// Constructor that enables peer bootstrap with the supplied tracker.
    pub fn new_with_bootstrap(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
        bootstrap: Arc<BootstrapTracker>,
    ) -> Arc<Self> {
        Self::new_with_mode(http, block_size_oracle, bootstrap, true)
    }

    /// Discovers worker hash metadata only: seeds the shared [`BlockSizeOracle`]
    /// but neither subscribes to KV events nor maintains the local tree, because
    /// an external Indexer is the routing signal.
    pub fn new_metadata_only_with_http_and_oracle(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Arc<Self> {
        Self::new_with_mode(
            http,
            block_size_oracle,
            Arc::new(BootstrapTracker::disabled()),
            false,
        )
    }

    fn new_with_mode(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
        bootstrap: Arc<BootstrapTracker>,
        maintain_tree: bool,
    ) -> Arc<Self> {
        let tree = Arc::new(HashTree::new());
        let (tx, rx) = mpsc::channel::<WorkerEvent>(EVENT_CHANNEL_BUFFER);
        let (ctrl_tx, ctrl_rx) = mpsc::channel::<PumpControl>(16);
        let subscribers = Arc::new(KvEventSubscriberRegistry::new(tx.clone()));
        let load_subscribers = Arc::new(KvEventSubscriberRegistry::with_kind(tx, SubKind::Load));
        let engine_reported_load = EngineReportedLoadTable::new();
        let cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>> = Arc::new(Mutex::new(HashMap::new()));
        let live_workers: Arc<Mutex<HashSet<KvWorkerId>>> = Arc::new(Mutex::new(HashSet::new()));
        let pump_cancel = CancellationToken::new();
        let peers = Arc::new(PeerRegistry::new());
        let tally = Arc::new(EventTally::new());
        let pump = tokio::spawn(pump_loop(
            PumpDeps {
                tally: Arc::clone(&tally),
                tree: tree.clone(),
                engine_reported_load: engine_reported_load.clone(),
                cursors: cursors.clone(),
                live_workers: live_workers.clone(),
                bootstrap: Arc::clone(&bootstrap),
            },
            pump_cancel.clone(),
            rx,
            ctrl_rx,
        ));
        Arc::new(Self {
            tree,
            maintain_tree,
            subscribers,
            load_subscribers,
            engine_reported_load,
            pump: Mutex::new(Some(pump)),
            pump_cancel: pump_cancel.clone(),
            workers: Mutex::new(HashMap::new()),
            http,
            live_workers,
            cursors,
            tally,
            bootstrap,
            peers,
            ctrl_tx,
            snapshot_cache: Arc::new(AsyncMutex::new(None)),
            block_size_oracle,
        })
    }

    /// Shared handle to the bootstrap tracker. `/readyz` reads it to decide
    /// whether initial bootstrap has settled; the metrics surface reads it for
    /// the per-rank state gauge.
    pub fn bootstrap(&self) -> Arc<BootstrapTracker> {
        Arc::clone(&self.bootstrap)
    }

    /// Shared accessor for the per-process block-size oracle.
    pub fn block_size_oracle(&self) -> Arc<BlockSizeOracle> {
        Arc::clone(&self.block_size_oracle)
    }

    /// Clone the underlying tree handle for cache-aware selection and
    /// metrics. The pump is the sole writer; callers should treat the
    /// returned handle as read-only.
    pub fn tree(&self) -> Arc<HashTree> {
        self.tree.clone()
    }

    /// Handles for the `/metrics` storage-tier series, or `None` when this
    /// router does not maintain a local tree.
    ///
    /// In metadata-only mode (an external Indexer is the routing signal) no KV
    /// subscription is opened, so every tier series would be a structural
    /// zero — while their own HELP text tells the operator to read a zero
    /// `CPU_PINNED` row as "the tier stream is not reaching the router". That
    /// is a different fault with a different fix, so emit nothing rather than
    /// a confidently wrong zero.
    pub fn metrics_source(&self) -> Option<KvIndexMetrics> {
        self.maintain_tree.then(|| KvIndexMetrics {
            tree: Arc::clone(&self.tree),
            tally: Arc::clone(&self.tally),
        })
    }

    /// Shared handle to the peer registry.
    pub fn peers(&self) -> Arc<PeerRegistry> {
        Arc::clone(&self.peers)
    }

    /// This replica as a bootstrap source. `None` in metadata-only mode,
    /// where no subscription is opened and the tree is always empty, so there
    /// is nothing to serve. Symmetric with [`Self::metrics_source`].
    pub fn snapshot_source(self: &Arc<Self>) -> Option<Arc<Self>> {
        self.maintain_tree.then(|| Arc::clone(self))
    }

    /// This replica's snapshot, encoded and ready to serve.
    ///
    /// Single-flighted behind an async lock so a boot herd shares one build,
    /// and a cached entry is reused while it meets `max_age` per
    /// [`meets_max_age`]. The build runs detached and owns the lock, so a
    /// caller that gives up mid-build does not cancel it.
    pub async fn peer_snapshot_body(self: &Arc<Self>, max_age: Duration) -> SnapshotBody {
        let requested_at = Instant::now();
        let mut cache = Arc::clone(&self.snapshot_cache).lock_owned().await;
        if let Some(c) = cache.as_ref() {
            if meets_max_age(c.exported_at, requested_at, max_age) {
                return c.body.clone();
            }
        }
        // Release the stale entry before building, so a fleet-sized body is
        // not held alongside its replacement.
        *cache = None;
        let this = Arc::clone(self);
        let build = tokio::spawn(async move {
            let (body, sampled_at) = this.build_peer_snapshot().await;
            if let Some(exported_at) = sampled_at {
                *cache = Some(CachedSnapshot {
                    exported_at,
                    body: body.clone(),
                });
            }
            body
        });
        match build.await {
            Ok(body) => body,
            Err(e) => {
                // The build task panicked or the runtime is shutting down;
                // nothing was cached, so a later request retries.
                warn!(error = %e, "kv-bootstrap: snapshot build task failed");
                SnapshotBody::default()
            }
        }
    }

    /// Walk, filter and encode one peer snapshot for [`Self::peer_snapshot_body`].
    ///
    /// Returns the body and, when it is worth caching, the instant its
    /// contents were sampled. An empty body means the encode failed; it is
    /// never cacheable, so the route's non-success answer is not pinned for a
    /// whole TTL.
    async fn build_peer_snapshot(&self) -> (SnapshotBody, Option<Instant>) {
        // See `CachedSnapshot::exported_at`.
        let exported_at = Instant::now();

        // Read the cursors BEFORE walking the tree, never after.
        //
        // The order is load-bearing: `export_snapshot` releases the tree
        // lock in chunks, so it is not one instant, and the pump mutates the
        // tree before advancing the cursor. Reading cursors last can therefore
        // report a sequence whose effects the walk only partially captured —
        // and the consumer would then filter its own copy of that batch as
        // already-reflected, losing a `BlockRemoved` permanently. Reading them
        // first makes the watermark lag the tree instead: the consumer replays
        // deltas the snapshot already has, and insert/remove are idempotent,
        // so it converges.
        let cursor_by_worker: Vec<(KvWorkerId, i64)> = self
            .cursors
            .lock()
            .iter()
            .map(|(w, seq)| (w.clone(), *seq))
            .collect();
        let tree = Arc::clone(&self.tree);
        let walked = tokio::task::spawn_blocking(move || tree.export_snapshot()).await;
        let (worker_table, nodes) = match walked {
            Ok(v) => v,
            Err(e) => {
                // The walk panicked or the runtime is shutting down. Answer
                // with a not-ready snapshot rather than a partial tree, and do
                // not cache it, so the next request makes a real attempt.
                // Encoding inline is fine: this body is a handful of bytes.
                warn!(error = %e, "kv-bootstrap: snapshot walk failed; reporting not-ready to peers");
                return (encode_snapshot(&self.not_ready_snapshot()), None);
            }
        };
        let index_of: HashMap<&KvWorkerId, u32> = worker_table
            .iter()
            .enumerate()
            .map(|(i, w)| (w, i as u32))
            .collect();
        // Re-check each cursor against the live map before reporting it. A
        // cursor only ever advances, so one that went backwards or vanished
        // during the walk means that rank's publisher reset, or the rank was
        // removed and re-added: the walk may hold blocks from the NEW stream
        // while the pre-walk cursor still names a position in the old one, and
        // a consumer seeded with it would filter the new stream's batches as
        // already applied. Dropping the cursor reads, to the consumer, as "the
        // producer never saw this rank", which runs it cold instead.
        let cursors: Vec<(u32, i64)> = {
            let live = self.cursors.lock();
            cursor_by_worker
                .iter()
                .filter(|(w, seq)| live.get(w).is_some_and(|now| now >= seq))
                .filter_map(|(w, seq)| index_of.get(w).map(|&i| (i, *seq)))
                .collect()
        };
        let has_nodes = !nodes.is_empty();
        let workers = worker_table.iter().map(WireWorker::from).collect();
        let snap = self.wire_snapshot(has_nodes, workers, cursors, nodes);
        // Encode on the blocking pool for the same reason the walk goes there:
        // serialising and compressing a fleet-sized tree is CPU-bound with no
        // await point, and this runs on the runtime that is also proxying
        // requests.
        match tokio::task::spawn_blocking(move || encode_snapshot(&snap)).await {
            Ok(body) => {
                let cacheable = (!body.is_empty()).then_some(exported_at);
                (body, cacheable)
            }
            Err(e) => {
                // Runtime shutting down or the encode panicked. Answer this
                // caller without caching, so a later request retries.
                warn!(error = %e, "kv-bootstrap: snapshot encode failed");
                (SnapshotBody::default(), None)
            }
        }
    }

    /// This replica's cursor table alone, encoded as JSON, with no tree.
    ///
    /// Reads the cursor map only: no tree walk, no cache. Every observed rank
    /// is reported, including ones that no longer carry nodes; the full export
    /// lists only carriers. `nodes` is always empty and `producer_ready`
    /// reflects the tree.
    pub fn peer_cursors_body(&self) -> Bytes {
        let (workers, cursors) = {
            let guard = self.cursors.lock();
            let mut workers: Vec<WireWorker> = Vec::with_capacity(guard.len());
            let mut cursors: Vec<(u32, i64)> = Vec::with_capacity(guard.len());
            for (i, (w, seq)) in guard.iter().enumerate() {
                workers.push(WireWorker::from(w));
                cursors.push((i as u32, *seq));
            }
            (workers, cursors)
        };
        let snap = self.wire_snapshot(self.tree.node_count() > 0, workers, cursors, Vec::new());
        // Encoded inline, unlike the tree export: this body is one entry per
        // rank, so a blocking-pool hop would cost more than the serialise.
        encode_json(&snap)
    }

    /// A [`PeerSnapshot`] stamped with this replica's hash config.
    ///
    /// `has_nodes` is whether the tree being described holds any nodes; see
    /// [`PeerSnapshot::producer_ready`].
    fn wire_snapshot(
        &self,
        has_nodes: bool,
        workers: Vec<WireWorker>,
        cursors: Vec<(u32, i64)>,
        nodes: Vec<SnapshotNode>,
    ) -> PeerSnapshot {
        let hash_config = self.block_size_oracle.hash_config();
        let (block_size, is_bigram) = hash_config.unwrap_or((0, false));
        PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size,
            is_bigram,
            producer_ready: hash_config.is_some() && self.bootstrap.settled() && has_nodes,
            workers,
            cursors,
            nodes,
        }
    }

    /// (test-only) Apply one stored block and its cursor directly, bypassing the pump.
    #[cfg(test)]
    pub(crate) fn seed_stored_block_for_test(
        &self,
        worker: &KvWorkerId,
        seq: i64,
        block_hash: i64,
    ) {
        self.tree
            .insert_tiered(worker, None, &[block_hash], Tiers::DEVICE);
        self.cursors.lock().insert(worker.clone(), seq);
    }

    /// (test-only) Record a cursor for a rank that carries no blocks.
    #[cfg(test)]
    pub(crate) fn seed_cursor_only_for_test(&self, worker: &KvWorkerId, seq: i64) {
        self.cursors.lock().insert(worker.clone(), seq);
    }

    /// A snapshot that declares itself not ready, for the paths that must
    /// answer without a tree.
    fn not_ready_snapshot(&self) -> PeerSnapshot {
        self.wire_snapshot(false, Vec::new(), Vec::new(), Vec::new())
    }

    /// Shared accessor for the engine-load table. Load values are written solely by the pump
    /// (from `LoadStat` events); `add_worker` / `remove_worker` here manage
    /// the expected set and per-worker eviction.
    pub fn engine_reported_load(&self) -> Arc<EngineReportedLoadTable> {
        Arc::clone(&self.engine_reported_load)
    }

    /// Register a worker. If `preresolved` is `Some`, the caller has
    /// already fetched `/server_info` (worker manager path) and we skip
    /// the internal HTTP round-trip; otherwise (standalone callers,
    /// e.g. integration tests) we fall back to `fetch_event_config`.
    ///
    /// Opens one ZMQ SUB per advertised DP rank for each usable stream. In
    /// metadata-only mode KV subscriptions remain disabled, but the separate
    /// #34608 load stream is still attached when its full descriptor exists.
    /// If the worker is not publishing KV events (older SGLang, opt-out
    /// config), this is a logged no-op — the worker still routes via the
    /// non-cache-aware policies.
    pub async fn add_worker(&self, worker_url: &str, preresolved: Option<EventConfig>) {
        let cfg: EventConfig = match preresolved {
            Some(c) => c,
            None => match fetch_event_config(worker_url, &self.http).await {
                Ok(Some(c)) => c,
                Ok(None) => {
                    info!(
                        worker_url = %worker_url,
                        "kv-events: worker is not publishing; cache-aware routing disabled for this worker",
                    );
                    return;
                }
                Err(e) => {
                    warn!(
                        worker_url = %worker_url,
                        error = %e,
                        "kv-events: /server_info introspection failed; skipping subscriber",
                    );
                    return;
                }
            },
        };
        // Reconcile this worker's `page_size` with the oracle BEFORE
        // any subscriber state is created. The first worker establishes
        // the value; later workers must agree. A mismatch means the
        // router and at least one engine would compute different block
        // hashes for the same prompt, silently destroying cache-aware
        // routing quality — reject loudly instead.
        if let Err(err) = self.block_size_oracle.try_set(cfg.block_size) {
            warn!(
                worker_url = %worker_url,
                established_block_size = err.established,
                worker_block_size = err.candidate,
                "kv-events: worker page_size disagrees with established block_size; \
                 skipping worker — cache-aware routing requires every worker to publish \
                 at the same block size",
            );
            return;
        }
        // Establish the bigram flag alongside block_size. EAGLE-family workers
        // hash KV blocks over token bigrams, so the policy must use the bigram
        // hasher for its query hashes to match the worker's stored hashes.
        self.block_size_oracle.set_bigram(cfg.is_bigram);
        let kv_dp_ranks = if self.maintain_tree {
            subscribable_ranks(cfg.port_base, cfg.dp_size)
        } else {
            Vec::new()
        };
        let load_descriptor_complete = cfg.load_port_base.is_some() && cfg.load_topic.is_some();
        if cfg.load_port_base.is_some() != cfg.load_topic.is_some() {
            warn!(
                worker_url = %worker_url,
                load_port_base = ?cfg.load_port_base,
                load_topic = ?cfg.load_topic,
                "kv-events: incomplete load descriptor; refusing load subscription"
            );
        }
        let load_dp_ranks = cfg
            .load_port_base
            .filter(|_| load_descriptor_complete)
            .map(|port_base| subscribable_ranks(port_base, cfg.dp_size))
            .unwrap_or_default();
        let mut dp_ranks = kv_dp_ranks.clone();
        dp_ranks.extend(load_dp_ranks.iter().copied());
        dp_ranks.sort_unstable();
        dp_ranks.dedup();
        if dp_ranks.is_empty() {
            warn!(
                worker_url = %worker_url,
                port_base = cfg.port_base,
                dp_size = cfg.dp_size,
                "kv-events: no usable KV or load publisher ranks; skipping worker",
            );
            return;
        }
        if self.maintain_tree {
            info!(
                worker_url = %worker_url,
                dp_size = cfg.dp_size,
                port_base = cfg.port_base,
                load_port_base = ?cfg.load_port_base,
                block_size = cfg.block_size,
                is_bigram = cfg.is_bigram,
                "kv-events: subscribing",
            );
        } else {
            info!(
                worker_url = %worker_url,
                dp_size = cfg.dp_size,
                load_port_base = ?cfg.load_port_base,
                block_size = cfg.block_size,
                is_bigram = cfg.is_bigram,
                "kv-events: external Indexer configured; subscribing only to engine load",
            );
        }
        // Mark every rank live BEFORE the subscriber starts so any event
        // it queues is accepted by the pump.
        {
            let mut live = self.live_workers.lock();
            for &rank in &dp_ranks {
                live.insert(KvWorkerId {
                    url: worker_url.to_string(),
                    dp_rank: rank,
                });
            }
        }
        // Register for bootstrap BEFORE the subscriber starts too, so the very
        // first batch is held back rather than applied ahead of the snapshot.
        // Ordering matters in the other direction as well: the subscription
        // must be live before the snapshot is fetched, so no delta can fall
        // into the gap between the peer's export and our first received batch.
        //
        // KV ranks only. A load-only rank publishes no KV stream, so there is
        // nothing to hold and no watermark to splice against; registering it
        // would leave an obligation no snapshot can ever discharge.
        let bootstrap_ranks: Vec<KvWorkerId> = kv_dp_ranks
            .iter()
            .map(|&rank| KvWorkerId::new(worker_url.to_string(), rank))
            .collect();
        let bootstrap_obligations: Vec<(KvWorkerId, u64)> =
            if self.peer_bootstrap_enabled() && !bootstrap_ranks.is_empty() {
                self.bootstrap.register(&bootstrap_ranks)
            } else {
                Vec::new()
            };
        self.workers.lock().insert(
            worker_url.to_string(),
            WorkerEntry {
                dp_ranks: dp_ranks.clone(),
            },
        );
        if self.maintain_tree && !kv_dp_ranks.is_empty() {
            self.subscribers.add_worker(worker_url, &cfg).await;
        }
        // Registered ranks hold their batches until an `ApplySnapshot` or
        // `AbandonBootstrap` names them. Every production constructor passes a
        // disabled tracker, which registers nothing.
        drop(bootstrap_obligations);
        // Mark only the ranks that have an actual SUB socket. `EngineReportedLoadTable`
        // then rejects missing or stale advertised ranks as a whole worker.
        if !load_dp_ranks.is_empty() {
            for rank in &load_dp_ranks {
                self.engine_reported_load
                    .mark_expected_rank(worker_url, *rank);
            }
            self.load_subscribers.add_worker(worker_url, &cfg).await;
        }
    }

    /// Whether this worker's ranks enter the bootstrap state machine. Keyed on
    /// configured, not settled, so a late worker still gets an obligation; and
    /// independent of visible peers, because registering arms the deadline.
    fn peer_bootstrap_enabled(&self) -> bool {
        self.bootstrap.enabled()
    }

    /// Tear down a worker's subscribers and clear it from the tree.
    /// Idempotent: a remove for a worker that was never added is a no-op.
    ///
    /// The live-worker entries are dropped **before** the subscriber join,
    /// so any event still buffered in the mpsc by the time the pump
    /// reaches it is dropped instead of re-inserted into the tree.
    pub async fn remove_worker(&self, worker_url: &str) {
        let Some(entry) = self.workers.lock().remove(worker_url) else {
            return;
        };
        let ids: Vec<KvWorkerId> = entry
            .dp_ranks
            .iter()
            .map(|&dp_rank| KvWorkerId {
                url: worker_url.to_string(),
                dp_rank,
            })
            .collect();
        // 1. Mark every rank dead. Any pump-queued events arriving after
        //    this point will be filtered.
        {
            let mut live = self.live_workers.lock();
            for id in &ids {
                live.remove(id);
            }
        }
        // 2. Cancel and join the per-rank subscriber tasks (KV + load). No
        //    further events for these ranks will be queued after this returns.
        self.subscribers.remove_worker(worker_url).await;
        self.load_subscribers.remove_worker(worker_url).await;
        // 3. Drop the worker's engine load and each rank's bootstrap state, so
        //    a rank that never finished cannot hold the readiness gate open.
        //    Any event already in the mpsc buffer at this point will be
        //    filtered by the live-set check inside the pump.
        self.engine_reported_load.forget_worker(worker_url);
        self.bootstrap.forget(&ids);
        // 4. Hand the per-rank teardown to the pump: `held`,
        //    `awaiting_splice_proof`, the tree carriers, and the cursor. Doing
        //    the tree/cursor half here would race a graft already past its
        //    gates (see the `ForgetRanks` arm).
        //
        //    `bootstrap.forget` above must precede this send: it is what makes
        //    an in-flight ApplySnapshot for these ranks a no-op.
        let (done_tx, done_rx) = oneshot::channel();
        let queued = self
            .ctrl_tx
            .send(PumpControl::ForgetRanks {
                ranks: ids.clone(),
                done: Some(done_tx),
            })
            .await
            .is_ok();
        // Wait, so a worker that flaps straight back cannot have its fresh
        // state wiped by this forget.
        if !queued || done_rx.await.is_err() {
            // Only reachable once the pump has exited, i.e. during shutdown.
            // Clean up inline so the state does not outlive the worker in that
            // case; there is no writer left to race.
            debug!("kv-events: pump is gone; tearing down worker state inline");
            let mut cursors = self.cursors.lock();
            for id in &ids {
                self.tree.clear_worker(id);
                cursors.remove(id);
            }
        }
    }

    /// Number of worker URLs the index is currently subscribed to. The
    /// count includes workers whose `/server_info` resolved but excludes
    /// any whose discovery returned `Ok(None)` (worker reachable but not
    /// publishing) or `Err` (transient discovery failure). Exposed for
    /// tests + future metrics; not part of the routing hot path.
    pub fn known_worker_count(&self) -> usize {
        self.workers.lock().len()
    }

    /// Shut down the pump task. Cancels the subscriber registry first so no
    /// further events are queued, then cancels the pump so any buffered
    /// events are discarded and the task exits promptly.
    pub async fn shutdown(&self) {
        self.subscribers.shutdown().await;
        self.load_subscribers.shutdown().await;
        self.pump_cancel.cancel();
        let handle = self.pump.lock().take();
        if let Some(h) = handle {
            // 2s ceiling guards against a pathological tokio runtime
            // teardown; under normal operation the pump exits within one
            // poll of `pump_cancel.cancelled()`.
            match tokio::time::timeout(Duration::from_secs(2), h).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => warn!(error = %e, "kv-events pump task did not join cleanly"),
                Err(_) => warn!("kv-events pump task did not stop within 2s"),
            }
        }
    }
}

/// Shared state the pump task reads and writes.
struct PumpDeps {
    tally: Arc<EventTally>,
    tree: Arc<HashTree>,
    engine_reported_load: Arc<EngineReportedLoadTable>,
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    bootstrap: Arc<BootstrapTracker>,
}

/// Drain `WorkerEvent`s: apply KV `Batch`es to the tree and `Load` snapshots
/// to the engine-load table. Out-of-order (seq ≤ last_applied) and stale
/// (worker not in `live_workers`) KV batches are skipped; `Load` is a gauge
/// with no seq. `PublisherReset` events clear the cursor so a publisher
/// restarting from seq=1 (after sending END_SEQ) is not filtered.
///
/// Also the sole writer of tree state, including snapshot grafts arriving as
/// [`PumpControl`]; see the single-writer property in [`super::tree`].
async fn pump_loop(
    deps: PumpDeps,
    cancel: CancellationToken,
    mut rx: mpsc::Receiver<WorkerEvent>,
    mut ctrl_rx: mpsc::Receiver<PumpControl>,
) {
    let PumpDeps {
        tally,
        tree,
        engine_reported_load,
        cursors,
        live_workers,
        bootstrap,
    } = deps;
    let pump_state = PumpState {
        tree: &tree,
        cursors: &cursors,
        tally: &tally,
        bootstrap: &bootstrap,
        live_workers: &live_workers,
    };

    // Batches held back while their rank is `Pending`. Pump-local: the pump is
    // the only task that touches it, so no lock is needed.
    let mut held: HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>> = HashMap::new();
    // Ranks grafted from a snapshot whose continuity with the live stream is
    // not yet provable, mapped to the watermark the first arriving batch must
    // not exceed by more than one.
    //
    // WHY deferred: a snapshot can be grafted before the rank's first live
    // batch has even arrived, so there is nothing to compare the watermark
    // against yet. The check runs on whichever batch turns up first — held or
    // live — and the entry is consumed by that one check.
    let mut awaiting_splice_proof: HashMap<KvWorkerId, i64> = HashMap::new();
    // Once the control channel closes its `recv()` resolves immediately and
    // forever, so it must be dropped from the select or the loop spins hot.
    let mut ctrl_open = true;

    loop {
        let ev = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                info!("kv-events pump: shutdown requested; exiting");
                return;
            }
            // Control first: releasing held batches unblocks readiness, and the
            // channel is near-empty in steady state.
            ctrl = ctrl_rx.recv(), if ctrl_open => {
                match ctrl {
                    Some(PumpControl::ApplySnapshot {
                        obligations,
                        vetted,
                    }) => {
                        apply_snapshot(
                            &pump_state,
                            &mut held,
                            &mut awaiting_splice_proof,
                            &obligations,
                            *vetted,
                        );
                    }
                    Some(PumpControl::ForgetRanks { ranks, done }) => {
                        for rank in ranks {
                            held.remove(&rank);
                            awaiting_splice_proof.remove(&rank);
                            // On the single writer so it is strictly ordered with
                            // any graft: a graft queued before this is undone here;
                            // one queued after fails its gates.
                            tree.clear_worker(&rank);
                            cursors.lock().remove(&rank);
                        }
                        if let Some(done) = done {
                            let _ = done.send(());
                        }
                    }
                    Some(PumpControl::AbandonBootstrap { obligations }) => {
                        for (rank, epoch) in obligations {
                            if !still_owed(&pump_state, &rank, epoch) {
                                continue;
                            }
                            fail_rank(
                                &pump_state,
                                &mut held,
                                &rank,
                                false,
                                RankOutcome::Abandoned,
                            );
                        }
                    }
                    None => {
                        debug!("kv-events pump: control channel closed");
                        ctrl_open = false;
                    }
                }
                continue;
            }
            recv = rx.recv() => match recv {
                Some(ev) => ev,
                None => {
                    warn!("kv-events pump: receiver closed unexpectedly; exiting");
                    return;
                }
            }
        };

        // Filter events from workers that are no longer attached. This is
        // load-bearing: `remove_worker` clears the live set BEFORE joining
        // the subscriber task, so any event still buffered when the pump
        // reaches it would otherwise re-pollute the tree.
        let worker = ev.worker();
        if !live_workers.lock().contains(worker) {
            debug!(
                worker = ?worker,
                "kv-events pump: dropping event from detached worker",
            );
            continue;
        }

        match ev {
            WorkerEvent::Load { worker, load } => {
                // Gauge: last value wins, no sequence/dedup. The live-worker
                // filter above already dropped load from detached workers.
                engine_reported_load.set(&worker.url, worker.dp_rank, load, Instant::now());
            }
            WorkerEvent::PublisherReset { worker } => {
                // A reset means the engine restarted with an empty cache and
                // renumbers from 1: any splice proof, grafted state or held
                // batch describes a cache that no longer exists, so drop all of
                // it and relearn from the new stream.
                if awaiting_splice_proof.remove(&worker).is_some() {
                    bootstrap.record_rank_outcome(RankOutcome::PublisherReset);
                }
                let state = bootstrap.state_of(&worker);
                tree.clear_worker(&worker);
                match state {
                    Some(BootstrapState::Recovered) => {
                        bootstrap.set(&worker, BootstrapState::Failed);
                    }
                    Some(BootstrapState::Pending) => {
                        warn!(
                            worker = ?worker,
                            "kv-bootstrap: publisher reset while awaiting a snapshot; \
                             abandoning bootstrap for this rank",
                        );
                        fail_rank(
                            &pump_state,
                            &mut held,
                            &worker,
                            true,
                            RankOutcome::PublisherReset,
                        );
                    }
                    _ => {}
                }
                if cursors.lock().remove(&worker).is_some() {
                    info!(
                        worker = ?worker,
                        "kv-events pump: publisher reset; cursor cleared",
                    );
                }
            }
            WorkerEvent::Batch {
                worker,
                seq,
                mut batch,
            } => {
                // A rank still awaiting its snapshot holds its batches: applying
                // them first would put live deltas *under* the snapshot, where a
                // stale `BlockStored` from the peer could resurrect a block this
                // rank has already evicted.
                if bootstrap.enabled()
                    && bootstrap.state_of(&worker) == Some(BootstrapState::Pending)
                {
                    if held.get(&worker).map_or(0, VecDeque::len) >= PENDING_BATCH_LIMIT {
                        // Dropping from the middle of the stream would leave a
                        // hole the snapshot cannot be spliced across, so give up
                        // on bootstrapping this rank and let it run live. The
                        // queue itself is intact (nothing was dropped to reach
                        // the cap), so it is replayed rather than discarded.
                        warn!(
                            worker = ?worker,
                            limit = PENDING_BATCH_LIMIT,
                            "kv-bootstrap: held-batch limit reached before a snapshot arrived; \
                             abandoning bootstrap for this rank",
                        );
                        fail_rank(
                            &pump_state,
                            &mut held,
                            &worker,
                            false,
                            RankOutcome::Overflow,
                        );
                    } else {
                        // The cap counts batches, not bytes, and `token_ids` is
                        // the unbounded part of a batch (one entry per prompt
                        // token) that `apply_batch` never reads. Shed it so a
                        // long-context fleet cannot hold gigabytes at boot.
                        for event in &mut batch.events {
                            if let KvCacheEvent::BlockStored(b) = event {
                                b.token_ids = Vec::new();
                            }
                        }
                        if let Some(queue) = held.get_mut(&worker) {
                            queue.push_back((seq, batch));
                        } else {
                            held.entry(worker.clone())
                                .or_default()
                                .push_back((seq, batch));
                        }
                        continue;
                    }
                    // Falls through: the rank is no longer Pending, so this
                    // batch is applied directly below.
                }
                // First batch after a graft proves — or disproves — that the
                // snapshot joins up with this rank's live stream.
                let watermark = if awaiting_splice_proof.is_empty() {
                    None
                } else {
                    awaiting_splice_proof.remove(&worker)
                };
                if let Some(watermark) = watermark {
                    if leaves_gap(seq, watermark) {
                        warn!(
                            worker = ?worker,
                            peer_cursor = watermark,
                            first_live_seq = seq,
                            "kv-bootstrap: sequence gap between snapshot and live stream; \
                             discarding snapshot state for this rank to avoid stale cache entries",
                        );
                        discard_graft(&pump_state, &worker, RankOutcome::Gap);
                    } else {
                        // The deferred check passed: this rank's grafted state is
                        // now proven continuous with its live stream, which is the
                        // point at which it counts as warm.
                        bootstrap.record_rank_outcome(RankOutcome::Warm);
                    }
                }
                apply_batch(&tree, &cursors, &tally, &worker, seq, &batch);
            }
        }
    }
}

/// The pump-owned handles the bootstrap helpers share.
struct PumpState<'a> {
    tree: &'a HashTree,
    cursors: &'a Mutex<HashMap<KvWorkerId, i64>>,
    tally: &'a EventTally,
    bootstrap: &'a BootstrapTracker,
    live_workers: &'a Mutex<HashSet<KvWorkerId>>,
}

/// Whether a live stream whose first batch is `first_seq` leaves a hole after
/// a snapshot watermarked at `watermark`.
///
/// The watermark is a peer's word, so it is not trusted to have a successor:
/// `i64::MAX` would overflow `watermark + 1`, and a stream can never join it.
fn leaves_gap(first_seq: i64, watermark: i64) -> bool {
    watermark.checked_add(1).is_none_or(|next| first_seq > next)
}

/// Apply one batch, honouring the cursor's out-of-order filter.
///
/// This is the single place tree deltas are written, whether the batch came
/// straight off the wire or out of a bootstrap hold-back queue — which is what
/// makes cursor seeding sufficient to reconcile a snapshot with the live stream.
fn apply_batch(
    tree: &HashTree,
    cursors: &Mutex<HashMap<KvWorkerId, i64>>,
    tally: &EventTally,
    worker: &KvWorkerId,
    seq: i64,
    batch: &KvEventBatch,
) {
    // Bound on its own line: an `if let` scrutinee would keep the guard alive
    // through the logging below.
    let prev = cursors.lock().get(worker).copied();
    if let Some(p) = prev {
        if seq <= p {
            debug!(
                worker = ?worker,
                seq,
                last_applied = p,
                "kv-events pump: out-of-order batch; skipping",
            );
            return;
        }
        // The publisher's seq is dense, so a jump is exactly the batches ZMQ
        // dropped at its high-water mark. A removal clears only its own tier,
        // so losing a block's last removal leaves the worker owning it until
        // the next AllBlocksCleared or teardown; only the sequence shows that
        // happened. The operator-visible signature is tree coverage above 1.
        let lost = (seq - p - 1) as u64;
        if lost > 0 {
            tally.record_lost_batches(lost);
            warn!(
                worker = ?worker,
                seq,
                last_applied = p,
                lost,
                "kv-events pump: sequence gap; batches were dropped in transit and the tree may hold stale tiers for this worker",
            );
        }
    }
    for event in &batch.events {
        match event {
            // The `medium` tag decides which tier a store lands on and which
            // tier a removal clears, so a device eviction leaves a worker that
            // still holds the block on host as an owner — see the tree's
            // "Storage tiers" docs.
            KvCacheEvent::BlockStored(b) => {
                tally.record(
                    EventKind::BlockStored,
                    b.medium.as_deref(),
                    b.block_hashes.len(),
                );
                tree.insert_tiered(
                    worker,
                    b.parent_block_hash,
                    &b.block_hashes,
                    Tiers::for_store(b.medium.as_deref()),
                );
            }
            KvCacheEvent::BlockRemoved(b) => {
                tally.record(
                    EventKind::BlockRemoved,
                    b.medium.as_deref(),
                    b.block_hashes.len(),
                );
                tree.remove_tiered(
                    worker,
                    &b.block_hashes,
                    Tiers::for_remove(b.medium.as_deref()),
                );
            }
            KvCacheEvent::AllBlocksCleared => {
                tally.record(EventKind::AllBlocksCleared, None, 0);
                tree.clear_worker(worker);
            }
        }
    }
    let mut guard = cursors.lock();
    if let Some(cursor) = guard.get_mut(worker) {
        *cursor = seq;
    } else {
        guard.insert(worker.clone(), seq);
    }
}

/// Give up on bootstrapping `rank`: release whatever it held and mark it
/// [`BootstrapState::Failed`].
///
/// `discard_held` is for a queue whose contents describe a cache that no
/// longer exists (a publisher reset): drop it and resume from the live stream.
/// Otherwise the intact queue is replayed as live deltas.
fn fail_rank(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    rank: &KvWorkerId,
    discard_held: bool,
    outcome: RankOutcome,
) {
    // Only transition ranks that are actually mid-bootstrap; an
    // AbandonBootstrap racing a successful ApplySnapshot must not undo it.
    //
    // This gate is also what keeps the rank tally once-per-rank: a second
    // attempt to fail an already-resolved rank returns before recording.
    if st.bootstrap.state_of(rank) != Some(BootstrapState::Pending) {
        held.remove(rank);
        return;
    }
    discard_graft(st, rank, outcome);
    let queue = held.remove(rank).unwrap_or_default();
    if discard_held {
        return;
    }
    for (seq, batch) in queue {
        apply_batch(st.tree, st.cursors, st.tally, rank, seq, &batch);
    }
}

/// Mark `rank` [`BootstrapState::Failed`], tally `outcome`, and drop
/// everything a snapshot contributed for it: tree carriers and cursor.
fn discard_graft(st: &PumpState<'_>, rank: &KvWorkerId, outcome: RankOutcome) {
    st.bootstrap.set(rank, BootstrapState::Failed);
    st.bootstrap.record_rank_outcome(outcome);
    st.tree.clear_worker(rank);
    st.cursors.lock().remove(rank);
}

/// Whether the obligation `(rank, epoch)` is still this pump's to resolve.
///
/// Three gates, each excluding a different way a control message can be
/// stale: wrong incarnation (worker removed and re-added while the fetch was
/// in flight), no longer live (removed outright — every other tree write in
/// the pump checks this too), and no longer `Pending` (something already
/// resolved it).
fn still_owed(st: &PumpState<'_>, rank: &KvWorkerId, epoch: u64) -> bool {
    st.bootstrap.epoch_of(rank) == Some(epoch)
        && st.live_workers.lock().contains(rank)
        && st.bootstrap.state_of(rank) == Some(BootstrapState::Pending)
}

/// Graft a vetted snapshot, seed cursors, then release held batches.
///
/// Runs on the pump so it is the sole tree writer for the duration.
fn apply_snapshot(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    awaiting_splice_proof: &mut HashMap<KvWorkerId, i64>,
    obligations: &[(KvWorkerId, u64)],
    mut vetted: VettedSnapshot,
) {
    let (tree, cursors, bootstrap) = (st.tree, st.cursors, st.bootstrap);
    // Ranks that left `Pending` while the fetch was in flight are already
    // applying live deltas; grafting older state beneath them is exactly the
    // stale splice this design refuses.
    // Derived from the ranks this bootstrap owns, NOT from the snapshot's worker
    // table: a rank the peer never mentioned still has to leave `Pending`, and it
    // does so below via the missing-cursor path.
    let pending: HashSet<KvWorkerId> = obligations
        .iter()
        .filter(|(w, epoch)| still_owed(st, w, *epoch))
        .map(|(w, _)| w.clone())
        .collect();
    vetted.retain_workers(&pending);

    let node_count = vetted.node_count();
    if pending.is_empty() {
        debug!("kv-bootstrap: snapshot has no still-pending ranks to graft; discarding");
        return;
    }
    info!(
        ranks = pending.len(),
        nodes = node_count,
        "kv-bootstrap: grafting peer snapshot",
    );
    if let Err(e) = vetted.graft_into(tree) {
        warn!(
            error = %e,
            nodes = node_count,
            "kv-bootstrap: snapshot rejected by the tree; affected ranks will run cold",
        );
        for rank in pending {
            fail_rank(st, held, &rank, false, RankOutcome::TreeRejected);
        }
        return;
    }

    for rank in pending {
        // No cursor means the peer was not tracking this rank, so its tree
        // slice has no watermark to splice against — the rank is cold.
        let Some(peer_cursor) = vetted.cursor_for(&rank) else {
            debug!(
                worker = ?rank,
                "kv-bootstrap: peer had no cursor for this rank; running cold",
            );
            fail_rank(st, held, &rank, false, RankOutcome::Uncovered);
            continue;
        };

        // Splice check: the stream must continue from the snapshot's watermark.
        // A hole means a delta was lost (ZMQ drops at its HWM), and a lost
        // `BlockRemoved` would leave a permanent false cache hit — so bail to
        // cold rather than graft over a gap.
        //
        // The evidence may not exist yet: nothing says a batch has arrived for
        // this rank by now. When the queue is empty the check is deferred to
        // whichever batch lands first (see `awaiting_splice_proof`).
        let first_held = held.get(&rank).and_then(|q| q.front()).map(|(seq, _)| *seq);
        match first_held {
            Some(first_seq) if leaves_gap(first_seq, peer_cursor) => {
                warn!(
                    worker = ?rank,
                    peer_cursor,
                    first_held_seq = first_seq,
                    "kv-bootstrap: sequence gap between snapshot and live stream; \
                     running cold to avoid stale cache entries",
                );
                fail_rank(st, held, &rank, false, RankOutcome::Gap);
                continue;
            }
            Some(_) => {}
            None => {
                awaiting_splice_proof.insert(rank.clone(), peer_cursor);
            }
        }
        // A held batch proves the splice now; an empty queue does not.
        let proven = first_held.is_some();

        // Seed the watermark, never backwards: a cursor already ahead of the
        // peer's means our live stream has outrun the snapshot.
        {
            let mut guard = cursors.lock();
            let entry = guard.entry(rank.clone()).or_insert(peer_cursor);
            *entry = (*entry).max(peer_cursor);
        }
        bootstrap.set(&rank, BootstrapState::Recovered);
        // The seeded cursor filters whatever the snapshot already reflects.
        for (seq, batch) in held.remove(&rank).unwrap_or_default() {
            apply_batch(tree, cursors, st.tally, &rank, seq, &batch);
        }
        // Tallied only once the splice is proven. Otherwise the deferred
        // first-batch check records the verdict, so each rank is tallied once.
        if proven {
            bootstrap.record_rank_outcome(RankOutcome::Warm);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::kv_events::wire::{BlockRemoved, BlockStored, KvEventBatch};
    use crate::state::load_monitor::engine_reported_load::LoadStat;

    fn worker_id(url: &str, rank: u32) -> KvWorkerId {
        KvWorkerId {
            url: url.into(),
            dp_rank: rank,
        }
    }

    fn batch(events: Vec<KvCacheEvent>) -> KvEventBatch {
        KvEventBatch {
            ts: 0.0,
            events,
            attn_dp_rank: None,
        }
    }

    // ---- bootstrap enablement vs. settlement ----

    /// A worker discovered after readiness opened must still be allowed to warm,
    /// so `enabled()` does not follow `settled()`.
    #[test]
    fn enabled_survives_settling_so_late_workers_still_bootstrap() {
        let tracker = BootstrapTracker::new(Duration::from_millis(1));
        let id = worker_id("http://w1:30000", 0);
        tracker.register(std::slice::from_ref(&id));
        std::thread::sleep(Duration::from_millis(5));
        assert!(tracker.settled(), "deadline expiry settles readiness");
        assert!(
            tracker.enabled(),
            "settling must NOT disable bootstrap for workers found later",
        );
    }

    /// The disabled case must stay distinguishable from the finished case, or a
    /// router with no `--kv-peer-selector` would start holding batches.
    #[test]
    fn disabled_tracker_is_not_enabled() {
        let t = BootstrapTracker::disabled();
        assert!(t.settled());
        assert!(
            !t.enabled(),
            "no selector configured ⇒ never register ranks"
        );
    }

    /// The remaining window saturates at zero once the deadline passes, while
    /// the configured timeout stays positive.
    #[test]
    fn time_remaining_saturates_to_zero_once_expired() {
        let tracker = BootstrapTracker::new(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(
            tracker.time_remaining(),
            Some(Duration::ZERO),
            "expired window must read as zero, not as None",
        );
        assert!(
            tracker.timeout() > Duration::ZERO,
            "the configured timeout must stay positive",
        );
    }

    /// Bundle of plumbing returned by `spawn_pump` so individual tests
    /// can destructure just the bits they need.
    struct PumpHarness {
        tree: Arc<HashTree>,
        engine_reported_load: Arc<EngineReportedLoadTable>,
        cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
        tally: Arc<EventTally>,
        #[allow(dead_code)]
        live_set: Arc<Mutex<HashSet<KvWorkerId>>>,
        #[allow(dead_code)]
        cancel: CancellationToken,
        tx: mpsc::Sender<WorkerEvent>,
        pump: JoinHandle<()>,
        ctrl_tx: mpsc::Sender<PumpControl>,
    }

    /// Build a tree + cursors + live-set wired through `pump_loop` with
    /// the given workers pre-marked live. Bootstrap is pre-settled, so batches
    /// are applied immediately — the behaviour every pre-bootstrap test asserts.
    fn spawn_pump(live: &[KvWorkerId]) -> PumpHarness {
        spawn_pump_with_bootstrap(live, Arc::new(BootstrapTracker::disabled()))
    }

    /// As `spawn_pump`, but with a caller-supplied tracker so bootstrap
    /// hold-back, splicing, and abandonment can be driven directly.
    fn spawn_pump_with_bootstrap(
        live: &[KvWorkerId],
        bootstrap: Arc<BootstrapTracker>,
    ) -> PumpHarness {
        let tree = Arc::new(HashTree::new());
        let engine_reported_load = EngineReportedLoadTable::new();
        let cursors = Arc::new(Mutex::new(HashMap::new()));
        let live_set: Arc<Mutex<HashSet<KvWorkerId>>> =
            Arc::new(Mutex::new(live.iter().cloned().collect()));
        let cancel = CancellationToken::new();
        let (tx, rx) = mpsc::channel(4);
        let (ctrl_tx, ctrl_rx) = mpsc::channel(4);
        let tally = Arc::new(EventTally::new());
        let pump = tokio::spawn(pump_loop(
            PumpDeps {
                tally: Arc::clone(&tally),
                tree: tree.clone(),
                engine_reported_load: engine_reported_load.clone(),
                cursors: cursors.clone(),
                live_workers: live_set.clone(),
                bootstrap: bootstrap.clone(),
            },
            cancel.clone(),
            rx,
            ctrl_rx,
        ));
        PumpHarness {
            tree,
            engine_reported_load,
            cursors,
            tally,
            live_set,
            cancel,
            tx,
            pump,
            ctrl_tx,
        }
    }

    // -----------------------------------------------------------------------
    // Bootstrap hold-back / splice / bail-to-cold
    // -----------------------------------------------------------------------

    fn stored(parent: Option<i64>, hashes: Vec<i64>) -> KvCacheEvent {
        KvCacheEvent::BlockStored(BlockStored {
            parent_block_hash: parent,
            block_hashes: hashes,
            token_ids: vec![],
            block_size: 64,
            lora_id: None,
            medium: None,
        })
    }

    /// The obligation set for `ranks` as registered in `tracker`, as
    /// `add_worker` gets it from `register`.
    fn obligations(tracker: &BootstrapTracker, ranks: &[KvWorkerId]) -> Vec<(KvWorkerId, u64)> {
        ranks
            .iter()
            .map(|r| {
                (
                    r.clone(),
                    tracker.epoch_of(r).expect("rank must be registered"),
                )
            })
            .collect()
    }

    /// A tracker with one rank registered and a deadline far enough out that it
    /// never fires mid-test.
    fn pending_tracker(ids: &[KvWorkerId]) -> Arc<BootstrapTracker> {
        let t = Arc::new(BootstrapTracker::new(Duration::from_secs(3600)));
        t.register(ids);
        t
    }

    /// A snapshot carrying chain [100, 200] for `id`, watermarked at `cursor`.
    fn vetted_for(id: &KvWorkerId, cursor: i64) -> VettedSnapshot {
        vetted_for_workers(&[id], cursor)
    }

    /// As `vetted_for`, but every node carries every listed worker — for the
    /// tests that need one snapshot to span several ranks.
    fn vetted_for_workers(ids: &[&KvWorkerId], cursor: i64) -> VettedSnapshot {
        let carriers: Vec<u32> = (0..ids.len() as u32).collect();
        VettedSnapshot::from_parts_for_test(
            ids.iter().map(|id| (*id).clone()).collect(),
            vec![
                SnapshotNode {
                    parent: None,
                    block_hash: 100,
                    workers: carriers.clone(),
                    tiers: vec![],
                },
                SnapshotNode {
                    parent: Some(0),
                    block_hash: 200,
                    workers: carriers,
                    tiers: vec![],
                },
            ],
            ids.iter().map(|id| ((*id).clone(), cursor)).collect(),
            0,
        )
    }

    /// While a rank is `Pending` its deltas must not reach the tree — applying
    /// them first would let a stale snapshot land on top of newer state.
    #[tokio::test]
    async fn pump_holds_batches_while_rank_pending() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump_with_bootstrap(
            std::slice::from_ref(&id),
            pending_tracker(std::slice::from_ref(&id)),
        );

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 6,
            batch: batch(vec![stored(None, vec![10, 20])]),
        })
        .await
        .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            h.tree.match_prefix(None, &[10, 20]).matched_blocks,
            0,
            "a held batch must not be applied",
        );
        assert!(h.cursors.lock().is_empty(), "no cursor for a held rank");
    }

    /// The splice: snapshot grafts, cursor seeds, then held deltas replay on
    /// top and the rank reports Recovered.
    #[tokio::test]
    async fn pump_splices_snapshot_then_drains_held_batches() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Live delta continues exactly where the snapshot's watermark stops.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 6,
            batch: batch(vec![stored(Some(200), vec![300])]),
        })
        .await
        .unwrap();
        // Let the pump hold the batch first; its biased select would otherwise
        // take the control message ahead of it and never hold anything.
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        // Snapshot state present.
        assert_eq!(h.tree.match_prefix(None, &[100, 200]).matched_blocks, 2);
        // Held delta applied on top of it.
        let m = h.tree.match_prefix(None, &[100, 200, 300]);
        assert_eq!(m.matched_blocks, 3, "held batch must extend the snapshot");
        assert!(m.workers().contains(&id));
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(6));
        assert!(tracker.settled());
    }

    /// A held batch the snapshot already reflects is filtered by the seeded
    /// cursor — the reuse that makes cursor seeding sufficient to reconcile.
    #[tokio::test]
    async fn pump_seeded_cursor_filters_already_reflected_batches() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // seq 5 EQUALS the snapshot's watermark: a redelivery of a batch the graft
        // already reflects. Equality is the boundary — `seq <= p` vs `seq < p` —
        // and re-applying it would re-insert blocks the producer may since have
        // removed.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 5,
            batch: batch(vec![stored(None, vec![777])]),
        })
        .await
        .unwrap();
        // Held first, so the seeded cursor filters it on the replay.
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            h.tree.match_prefix(None, &[777]).matched_blocks,
            0,
            "a batch the snapshot already reflects must be filtered",
        );
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(5));
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
    }

    /// A hole between the snapshot watermark and the live stream means a delta
    /// was lost. Grafting anyway could leave a permanently stale entry, so the
    /// rank drops to cold: snapshot state cleared, live deltas still applied.
    ///
    /// This is the *deferred* path — the snapshot is grafted before any batch
    /// for the rank has arrived, so continuity can only be judged later. The
    /// pump's biased select drains the control channel first, which makes this
    /// the ordering that occurs naturally.
    #[tokio::test]
    async fn pump_detects_deferred_sequence_gap_and_runs_cold() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Watermark 5, but the first live batch is seq 9 — 6..8 were lost.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        // A gapped rank is terminal.
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "snapshot state must be discarded for a gapped rank",
        );
        // The live stream still applies: cold, not broken.
        assert!(h.tree.match_prefix(None, &[42]).workers().contains(&id));
        // Every rank is terminal, so readiness no longer waits on this one.
        assert!(tracker.settled());
    }

    /// Same gap, detected on the *immediate* path: the batch is already held
    /// when the snapshot arrives, so the watermark can be checked at graft time.
    #[tokio::test]
    async fn pump_detects_held_queue_sequence_gap_and_runs_cold() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        // Let the pump receive and hold the batch before the snapshot lands.
        // The pump's biased select would otherwise take the control message
        // first, which is the deferred path covered by the test above.
        tokio::time::sleep(Duration::from_millis(50)).await;

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "snapshot state must be discarded for a gapped rank",
        );
        assert!(h.tree.match_prefix(None, &[42]).workers().contains(&id));
    }

    /// The gap boundary, both directions, on both the deferred and the held path.
    ///
    /// A one-batch hole is the likeliest ZMQ-HWM drop, and it is exactly the case
    /// an off-by-one in `seq > watermark + 1` would wave through.
    #[tokio::test]
    async fn pump_gap_boundary_is_exact() {
        // watermark 5: seq 6 continues cleanly, seq 7 means batch 6 was lost.
        for (first_seq, expect_kept) in [(6i64, true), (7i64, false)] {
            let id = worker_id("http://w1", 0);
            let tracker = pending_tracker(std::slice::from_ref(&id));
            let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

            h.ctrl_tx
                .send(PumpControl::ApplySnapshot {
                    obligations: obligations(&tracker, std::slice::from_ref(&id)),
                    vetted: Box::new(vetted_for(&id, 5)),
                })
                .await
                .unwrap();
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq: first_seq,
                batch: batch(vec![stored(None, vec![9_000 + first_seq])]),
            })
            .await
            .unwrap();
            drop(h.tx);
            drop(h.ctrl_tx);
            h.pump.await.unwrap();

            let kept = h
                .tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id);
            assert_eq!(
                kept, expect_kept,
                "deferred path, watermark 5, first live seq {first_seq}",
            );
        }

        // Same boundary on the immediate (already-held) path.
        for (first_seq, expect_kept) in [(6i64, true), (7i64, false)] {
            let id = worker_id("http://w1", 0);
            let tracker = pending_tracker(std::slice::from_ref(&id));
            let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq: first_seq,
                batch: batch(vec![stored(None, vec![8_000 + first_seq])]),
            })
            .await
            .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
            h.ctrl_tx
                .send(PumpControl::ApplySnapshot {
                    obligations: obligations(&tracker, std::slice::from_ref(&id)),
                    vetted: Box::new(vetted_for(&id, 5)),
                })
                .await
                .unwrap();
            drop(h.tx);
            drop(h.ctrl_tx);
            h.pump.await.unwrap();

            let kept = h
                .tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id);
            assert_eq!(
                kept, expect_kept,
                "held path, watermark 5, first held seq {first_seq}",
            );
        }
    }

    /// A contiguous live batch after a graft must NOT be mistaken for a gap.
    #[tokio::test]
    async fn pump_contiguous_batch_after_graft_keeps_snapshot() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        // seq 6 continues directly from watermark 5.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 6,
            batch: batch(vec![stored(Some(200), vec![300])]),
        })
        .await
        .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        let m = h.tree.match_prefix(None, &[100, 200, 300]);
        assert_eq!(
            m.matched_blocks, 3,
            "snapshot must survive a contiguous delta"
        );
        assert!(m.workers().contains(&id));
    }

    /// No peer could supply a snapshot: held deltas must still be released, or
    /// the rank would buffer forever and never settle.
    #[tokio::test]
    async fn pump_abandon_bootstrap_releases_held_batches() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 3,
            batch: batch(vec![stored(None, vec![55, 66])]),
        })
        .await
        .unwrap();
        // Held first, so it is the abandon's replay that releases it.
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        let m = h.tree.match_prefix(None, &[55, 66]);
        assert_eq!(
            m.matched_blocks, 2,
            "held batches must be released, not lost"
        );
        assert!(m.workers().contains(&id));
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(3));
        assert!(tracker.settled());
    }

    /// A rank the peer's snapshot never mentions must still leave `Pending`, or
    /// it buffers every batch and routes cache-blind indefinitely.
    #[tokio::test]
    async fn pump_snapshot_fails_ranks_the_peer_did_not_cover() {
        let covered = worker_id("http://w1", 0);
        let uncovered = worker_id("http://w2", 0);
        let tracker = pending_tracker(&[covered.clone(), uncovered.clone()]);
        let h = spawn_pump_with_bootstrap(&[covered.clone(), uncovered.clone()], tracker.clone());

        // The uncovered rank has a batch held back; it must be released.
        h.tx.send(WorkerEvent::Batch {
            worker: uncovered.clone(),
            seq: 3,
            batch: batch(vec![stored(None, vec![555])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        // Snapshot covers only `covered`, but the bootstrap owns both ranks.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, &[covered.clone(), uncovered.clone()]),
                vetted: Box::new(vetted_for(&covered, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&covered), Some(BootstrapState::Recovered));
        assert_eq!(
            tracker.state_of(&uncovered),
            Some(BootstrapState::Failed),
            "a rank absent from the snapshot must not stay Pending",
        );
        // Its held batch is released rather than stranded.
        assert!(h
            .tree
            .match_prefix(None, &[555])
            .workers()
            .contains(&uncovered));
        assert!(tracker.settled(), "both ranks terminal ⇒ readiness opens");
    }

    /// A snapshot fetched for a PREVIOUS incarnation must not graft: it would
    /// seed a watermark from the old publisher's numbering, and every batch
    /// from the fresh publisher (restarting at seq 1) would then be filtered
    /// as out-of-order while the rank reports Recovered.
    #[tokio::test]
    async fn pump_snapshot_from_a_stale_incarnation_is_discarded() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let stale = obligations(&tracker, std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // The worker goes away and comes back: a new incarnation, Pending again.
        tracker.forget(std::slice::from_ref(&id));
        let fresh = tracker.register(std::slice::from_ref(&id));
        assert_ne!(stale[0].1, fresh[0].1, "test needs a new incarnation");

        // The old task's result finally arrives.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: stale,
                vetted: Box::new(vetted_for(&id, 5000)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "a previous incarnation's snapshot must not be grafted",
        );
        assert_eq!(
            h.cursors.lock().get(&id).copied(),
            None,
            "no stale watermark may be seeded, or the fresh publisher is filtered out",
        );
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
    }

    /// A graft must respect `live_workers` like every other tree write in the
    /// pump: a worker removed while the fetch was in flight must not get carriers.
    #[tokio::test]
    async fn pump_snapshot_skips_ranks_no_longer_live() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let obs = obligations(&tracker, std::slice::from_ref(&id));
        // Spawned with an EMPTY live set: the worker is gone.
        let h = spawn_pump_with_bootstrap(&[], tracker.clone());

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obs,
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "a deregistered worker must not gain carriers from a graft",
        );
        assert!(h.cursors.lock().is_empty());
    }

    /// A snapshot that arrives after its rank already went Failed must not be
    /// grafted under the live stream it lost the race to.
    #[tokio::test]
    async fn pump_snapshot_skips_ranks_no_longer_pending() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
            })
            .await
            .unwrap();
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "a late snapshot must not graft onto an already-live rank",
        );
    }

    /// Overflowing the hold-back queue abandons bootstrap, and the rank resumes
    /// live WITHOUT losing what it held: the queue reached the cap intact, so
    /// discarding it would drop every block stored while waiting — blocks the
    /// engine never re-announces.
    #[tokio::test]
    async fn pump_overflowing_held_queue_abandons_bootstrap() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        for seq in 1..=(PENDING_BATCH_LIMIT as i64 + 2) {
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch: batch(vec![stored(None, vec![seq])]),
            })
            .await
            .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        // The held prefix is replayed, then the batch that tripped the limit
        // and everything after it apply directly.
        let last = PENDING_BATCH_LIMIT as i64 + 2;
        for seq in [1, PENDING_BATCH_LIMIT as i64, last] {
            assert!(
                h.tree.match_prefix(None, &[seq]).workers().contains(&id),
                "batch {seq} must be applied across the overflow",
            );
        }
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(last));
        assert_eq!(
            h.tally.batches_lost(),
            0,
            "the replay is contiguous, so no sequence gap may be recorded",
        );
        assert!(tracker.settled());
    }

    // -----------------------------------------------------------------------
    // Bootstrap state-machine contract, as an executable property
    //
    // A randomised driver over the real pump checks the two contract
    // properties across interleavings.
    //
    // The contract:
    //   T (terminality) — once the bootstrap for a rank is resolved, that rank
    //     is never left `Pending`. A `Pending` rank buffers its events forever
    //     and holds the readiness gate, so this is the load-bearing property.
    //   L (latch) — `settled()` never transitions true -> false. Violating it
    //     turns a routine scale-up into a 503 on a serving replica.
    //
    // Cursor monotonicity is deliberately NOT asserted here: `fail_rank`
    // legitimately REMOVES a cursor, so a later lower value is correct after a
    // bail-to-cold. It is covered by the targeted tests above instead.
    // -----------------------------------------------------------------------

    /// Randomised interleavings of every control-plane and data-plane event,
    /// checked against the contract above.
    #[tokio::test]
    async fn property_bootstrap_contract_holds_under_random_interleavings() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        for seed in 0..96u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let n_workers = rng.gen_range(1..=3);
            let ranks: Vec<KvWorkerId> = (0..n_workers)
                .flat_map(|w| {
                    let dp = rng.gen_range(1..=2);
                    (0..dp)
                        .map(move |r| worker_id(&format!("http://w{w}"), r))
                        .collect::<Vec<_>>()
                })
                .collect();

            let tracker = pending_tracker(&ranks);
            let h = spawn_pump_with_bootstrap(&ranks, tracker.clone());

            let mut latched = false;
            let mut next_seq: HashMap<KvWorkerId, i64> = HashMap::new();
            // Ranks named in some control message's obligation set. Only these
            // are required to be terminal at the end — a rank nobody ever
            // resolved is legitimately still Pending until the deadline.
            let mut discharged: HashSet<KvWorkerId> = HashSet::new();
            // Set when a scale-up rank is registered, which legitimately leaves
            // an undischarged Pending entry behind.
            let mut registered_extra = false;

            for _ in 0..rng.gen_range(4..40) {
                let rank = ranks[rng.gen_range(0..ranks.len())].clone();
                match rng.gen_range(0..6) {
                    5 => {
                        // Scale-up: a brand-new rank appears mid-flight. This is
                        // what makes property L non-vacuous — without a fresh
                        // Pending entry after settlement, a missing latch could
                        // never be observed.
                        registered_extra = true;
                        tracker.register(&[worker_id(&format!("http://scaleup{seed}"), 0)]);
                    }
                    0 => {
                        // A live batch, sometimes with a deliberate seq gap.
                        let seq = next_seq.entry(rank.clone()).or_insert(1);
                        *seq += rng.gen_range(1..=3);
                        let s = *seq;
                        let _ =
                            h.tx.send(WorkerEvent::Batch {
                                worker: rank,
                                seq: s,
                                batch: batch(vec![stored(None, vec![s * 1000 + seed as i64])]),
                            })
                            .await;
                    }
                    1 => {
                        // The obligation set and the snapshot's coverage are
                        // generated INDEPENDENTLY. That decoupling is the whole
                        // point: a peer routinely knows nothing about some rank
                        // this replica registered, and the resulting
                        // "obligation without coverage" case is where the
                        // terminality property actually bites.
                        let obligation_ranks: Vec<KvWorkerId> = ranks
                            .iter()
                            .filter(|_| rng.gen_bool(0.7))
                            .cloned()
                            .collect();
                        let obligation_ranks = if obligation_ranks.is_empty() {
                            vec![rank.clone()]
                        } else {
                            obligation_ranks
                        };
                        // Covers an unrelated rank, or nothing at all.
                        let covered = if rng.gen_bool(0.3) {
                            None
                        } else {
                            Some(ranks[rng.gen_range(0..ranks.len())].clone())
                        };
                        let vetted = match covered {
                            Some(c) => vetted_for(&c, rng.gen_range(0..8)),
                            None => VettedSnapshot::from_parts_for_test(vec![], vec![], vec![], 0),
                        };
                        discharged.extend(obligation_ranks.iter().cloned());
                        let _ = h
                            .ctrl_tx
                            .send(PumpControl::ApplySnapshot {
                                obligations: obligations(&tracker, &obligation_ranks),
                                vetted: Box::new(vetted),
                            })
                            .await;
                    }
                    2 => {
                        let _ = h
                            .ctrl_tx
                            .send(PumpControl::AbandonBootstrap {
                                obligations: obligations(&tracker, &[rank]),
                            })
                            .await;
                    }
                    3 => {
                        let _ = h
                            .ctrl_tx
                            .send(PumpControl::ForgetRanks {
                                ranks: vec![rank],
                                done: None,
                            })
                            .await;
                    }
                    _ => {
                        let _ =
                            h.tx.send(WorkerEvent::PublisherReset { worker: rank })
                                .await;
                    }
                }

                // L: sampled continuously, because the violation is a transition.
                let settled_now = tracker.settled();
                let regressed = latched && !settled_now;
                assert!(!regressed, "seed {seed}: settled() went true -> false");
                latched |= settled_now;
            }

            drop(h.tx);
            drop(h.ctrl_tx);
            h.pump.await.unwrap();

            // T: every rank some control message took responsibility for must be
            // terminal. Deliberately NO blanket AbandonBootstrap first — that
            // would satisfy this no matter what the ApplySnapshot handler did.
            //
            // `None` is accepted: ForgetRanks models the worker being removed.
            for r in &discharged {
                match tracker.state_of(r) {
                    None => {}
                    Some(state) => assert!(
                        state.is_terminal(),
                        "seed {seed}: rank {r:?} was named in an obligation set but left \
                         Pending — it would buffer forever and hold the readiness gate",
                    ),
                }
            }
            if discharged.len() == ranks.len() && !registered_extra {
                assert!(
                    tracker.settled(),
                    "seed {seed}: every rank resolved but the readiness gate never opened",
                );
            }
        }
    }

    /// Count of one rank-outcome label, for the exactly-once assertions below.
    fn rank_count(tracker: &BootstrapTracker, label: &str) -> u64 {
        tracker
            .rank_outcome_counts()
            .into_iter()
            .find(|(l, _)| *l == label)
            .map(|(_, c)| c)
            .unwrap_or(0)
    }

    /// Graft a rank with NOTHING held, so its splice proof is deferred, and leave
    /// it in that state. The shape the deferred-proof tests start from.
    async fn graft_with_deferred_proof(
        id: &KvWorkerId,
        watermark: i64,
    ) -> (Arc<BootstrapTracker>, PumpHarness) {
        let tracker = pending_tracker(std::slice::from_ref(id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(id), tracker.clone());
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(id)),
                vetted: Box::new(vetted_for(id, watermark)),
            })
            .await
            .unwrap();
        (tracker, h)
    }

    /// A grafted-but-unproven rank must NOT be counted warm yet: the whole point
    /// of the split counter is that `warm` means "proven", so counting at graft
    /// time and demoting later would double-count under two labels.
    #[tokio::test]
    async fn pump_deferred_proof_is_not_counted_warm_until_proven() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        // Drain: the graft is handled before the channels close.
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert_eq!(
            rank_count(&tracker, "warm"),
            0,
            "an unproven splice must not be tallied warm",
        );
    }

    /// The contiguous batch that proves the splice is what tallies the rank warm
    /// — exactly once.
    #[tokio::test]
    async fn pump_deferred_proof_counts_warm_once_when_the_stream_joins_up() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        for seq in [6, 7, 8] {
            h.tx.send(WorkerEvent::Batch {
                worker: id.clone(),
                seq,
                batch: batch(vec![stored(None, vec![seq])]),
            })
            .await
            .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert_eq!(
            rank_count(&tracker, "warm"),
            1,
            "three batches after the proof must still tally one warm rank",
        );
        assert!(
            h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "a proven splice keeps its grafted state",
        );
    }

    /// Peer-attempt and per-rank tallies must live in separate counters: one
    /// accepted fetch can settle several ranks, so a shared counter would be
    /// divisible by nothing.
    #[tokio::test]
    async fn rank_and_peer_outcome_tallies_are_separate() {
        let a = worker_id("http://w1", 0);
        let b = worker_id("http://w2", 0);
        let tracker = pending_tracker(&[a.clone(), b.clone()]);
        let h = spawn_pump_with_bootstrap(&[a.clone(), b.clone()], tracker.clone());
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: obligations(&tracker, &[a.clone(), b.clone()]),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(rank_count(&tracker, "abandoned"), 2, "one count per rank");
        assert!(
            tracker.peer_outcome_counts().is_empty(),
            "no peer was contacted, so the peer counter must stay empty",
        );
    }

    /// A restarted publisher renumbers from 1, so its cursor MUST be cleared.
    ///
    /// Without this, every post-restart batch has `seq < last_applied` and is
    /// dropped as out-of-order: the rank's tree freezes at pre-restart state
    /// forever while every metric reports healthy.
    #[tokio::test]
    async fn pump_publisher_reset_clears_cursor_so_restarted_stream_applies() {
        let id = worker_id("http://w1", 0);
        // Settled tracker: this is the steady-state path, not bootstrap.
        let h = spawn_pump(std::slice::from_ref(&id));

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![11])]),
        })
        .await
        .unwrap();
        h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
            .await
            .unwrap();
        // A fresh publisher's first batch.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: batch(vec![stored(None, vec![22])]),
        })
        .await
        .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert!(
            h.tree.match_prefix(None, &[22]).workers().contains(&id),
            "a restarted publisher's stream must apply, not be filtered as stale",
        );
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(1));
        // The reset also clears pre-restart state, on this bootstrap-disabled
        // path too: the restarted engine holds none of those blocks.
        assert!(
            !h.tree.match_prefix(None, &[11]).workers().contains(&id),
            "pre-reset blocks must not survive as false cache hits",
        );
    }

    /// A reset while still Pending must bail the rank to cold.
    ///
    /// Post-reset seq 1 can never exceed the peer's watermark, so the gap check
    /// passes trivially, the stale tree is kept, and every real delta is filtered.
    #[tokio::test]
    async fn pump_publisher_reset_while_pending_abandons_bootstrap() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 900,
            batch: batch(vec![stored(None, vec![33])]),
        })
        .await
        .unwrap();
        h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Failed),
            "a reset mid-bootstrap must abandon, not wait for a snapshot it can no \
             longer splice",
        );
        assert!(tracker.settled());
    }

    /// A terminal rank is immutable. An Abandon arriving after a successful
    /// graft must not wipe the rank's tree and demote it to Failed.
    #[tokio::test]
    async fn pump_abandon_after_successful_graft_does_not_undo_it() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let obs = obligations(&tracker, std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obs.clone(),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        // A second bootstrap task for the same obligation set gives up.
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap { obligations: obs })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Recovered),
            "a late Abandon must not demote an already-Recovered rank",
        );
        assert!(
            h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "a late Abandon must not wipe a grafted tree",
        );
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(5));
    }

    /// The stale-incarnation guard on the ABANDON arm, mirroring the one
    /// already covered on the ApplySnapshot arm.
    #[tokio::test]
    async fn pump_abandon_from_a_stale_incarnation_is_ignored() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let stale = obligations(&tracker, std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Remove + re-add: a new incarnation, Pending again.
        tracker.forget(std::slice::from_ref(&id));
        let fresh = tracker.register(std::slice::from_ref(&id));
        assert_ne!(stale[0].1, fresh[0].1);

        // The previous incarnation's task gives up.
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap { obligations: stale })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "a stale Abandon must not force the new incarnation cold",
        );
    }

    /// `retain_workers` must strip carriers for ranks that are not pending,
    /// or a rank already applying live deltas gets peer state grafted UNDERNEATH
    /// it — the stale splice the design refuses.
    #[tokio::test]
    async fn pump_snapshot_does_not_graft_carriers_for_non_pending_ranks() {
        let pending_rank = worker_id("http://w1", 0);
        let live_rank = worker_id("http://w2", 0);
        let tracker = pending_tracker(&[pending_rank.clone(), live_rank.clone()]);
        let obs = obligations(&tracker, &[pending_rank.clone(), live_rank.clone()]);
        let h =
            spawn_pump_with_bootstrap(&[pending_rank.clone(), live_rank.clone()], tracker.clone());

        // `live_rank` leaves Pending first: it is now applying live deltas.
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: obligations(&tracker, std::slice::from_ref(&live_rank)),
            })
            .await
            .unwrap();
        // A snapshot covering BOTH ranks then arrives: every node carries both
        // worker-table slots.
        let vetted = vetted_for_workers(&[&pending_rank, &live_rank], 5);
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obs,
                vetted: Box::new(vetted),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        let m = h.tree.match_prefix(None, &[100, 200]);
        assert!(
            m.workers().contains(&pending_rank),
            "the still-pending rank must be grafted",
        );
        assert!(
            !m.workers().contains(&live_rank),
            "a rank already applying live deltas must NOT receive peer carriers",
        );
    }

    /// `ForgetRanks` must drop the pump-local queue and splice proof, or a
    /// re-added worker inherits the dead incarnation's high sequence numbers and
    /// its fresh stream reads as a permanent gap.
    #[tokio::test]
    async fn pump_forget_ranks_drops_held_state_and_tree() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Grafted tree state and a deferred splice proof to be torn down.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        h.ctrl_tx
            .send(PumpControl::ForgetRanks {
                ranks: vec![id.clone()],
                done: None,
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "ForgetRanks must clear the worker's carriers on the pump",
        );
        assert!(
            h.cursors.lock().get(&id).is_none(),
            "a stale cursor would filter the re-added worker's fresh stream",
        );
    }

    /// `ForgetRanks` drops `held` itself.
    ///
    /// A leaked queue is only observable through its EFFECT on the next
    /// incarnation — the dead incarnation's high sequence numbers sit at the front
    /// of the queue, so the fresh publisher's low watermark reads as a permanent
    /// gap and the re-added worker can never bootstrap.
    #[tokio::test]
    async fn pump_forget_ranks_drops_held_queue_so_readd_can_bootstrap() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Old incarnation holds a high-sequence batch.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 900,
            batch: batch(vec![stored(None, vec![77])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Worker removed, then re-added: fresh incarnation, publisher renumbered.
        h.ctrl_tx
            .send(PumpControl::ForgetRanks {
                ranks: vec![id.clone()],
                done: None,
            })
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        let fresh = tracker.register(std::slice::from_ref(&id));

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: batch(vec![stored(Some(200), vec![300])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: fresh,
                vetted: Box::new(vetted_for(&id, 0)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        // With a leaked queue, front() is seq 900 against watermark 0 -> spurious
        // gap -> Failed and the graft discarded.
        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Recovered),
            "a leaked held queue makes the re-added worker read as gapped forever",
        );
        assert!(h
            .tree
            .match_prefix(None, &[100, 200])
            .workers()
            .contains(&id));
    }

    /// A publisher reset means the engine came back with an empty cache, so
    /// the grafted blocks are now false cache hits that nothing will ever
    /// evict — the `BlockRemoved` for them was the old publisher's to send.
    /// And the dropped splice proof has to be tallied, or the rank's verdict
    /// never lands and `bootstrap_rank_total` stops summing to the ranks that
    /// finished.
    #[tokio::test]
    async fn pump_publisher_reset_after_a_graft_clears_the_tree_and_tallies() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), Arc::clone(&tracker));

        // Graft with no held batch, so the splice proof is deferred.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 7)),
            })
            .await
            .unwrap();
        h.tx.send(WorkerEvent::PublisherReset { worker: id.clone() })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "a restarted engine holds none of the grafted blocks",
        );
        assert_eq!(
            rank_count(&tracker, RankOutcome::PublisherReset.as_label()),
            1,
            "the discarded proof must land a verdict",
        );
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
    }

    /// Direct test of the pump loop's tree application — no sockets.
    #[tokio::test]
    async fn pump_applies_block_stored_to_tree() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tree, tx, pump) = (h.tree, h.tx, h.pump);

        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: batch(vec![KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![10, 20, 30],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: None,
            })]),
        })
        .await
        .unwrap();
        drop(tx);
        // Don't cancel — let rx.recv() return None naturally so any
        // queued events drain first. (The pump's `biased` select would
        // otherwise preempt unprocessed events on cancel.)
        pump.await.unwrap();

        let m = tree.match_prefix(None, &[10, 20, 30]);
        assert_eq!(m.matched_blocks, 3);
        assert!(m.holds(&id), "tree must hold the worker");
    }

    /// The pump must carry each event's `medium` into the tree. The engine's
    /// write-back sequence for a backed-up block is a host-tagged store
    /// followed by a device-tagged removal; applied tier-blind, the removal
    /// erased the worker and every repeat of that prefix routed cold for the
    /// whole host retention horizon.
    #[tokio::test]
    async fn pump_keeps_host_backed_block_owned_across_device_eviction() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tree, tx, pump) = (h.tree, h.tx, h.pump);

        let stored = |medium: Option<&str>| {
            KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![10, 20],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: medium.map(str::to_owned),
            })
        };
        let removed = |medium: Option<&str>| {
            KvCacheEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![20],
                medium: medium.map(str::to_owned),
            })
        };
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: batch(vec![
                stored(Some("GPU")),
                stored(Some("CPU_PINNED")),
                removed(Some("GPU")),
            ]),
        })
        .await
        .unwrap();
        drop(tx);
        pump.await.unwrap();

        let m = tree.match_prefix(None, &[10, 20]);
        assert_eq!(m.matched_blocks, 2, "host copy keeps the block routable");
        assert!(m.holds(&id));
        assert!(!m.device_workers().contains(&id), "device copy is gone");
    }

    /// The metadata-only gate. Its whole justification is that a structural
    /// zero would be read as "the tier stream is not reaching the router" — a
    /// different fault with a different fix — so the gate itself needs pinning:
    /// inverting it leaves every test green while `/metrics` starts lying.
    #[tokio::test]
    async fn metrics_source_is_none_only_without_a_local_tree() {
        let http = reqwest::Client::builder().build().unwrap();
        let with_tree =
            KvEventIndex::new_with_http_and_oracle(http.clone(), BlockSizeOracle::new());
        assert!(
            with_tree.metrics_source().is_some(),
            "a router maintaining its own tree must publish the tier series",
        );
        let metadata_only =
            KvEventIndex::new_metadata_only_with_http_and_oracle(http, BlockSizeOracle::new());
        assert!(
            metadata_only.metrics_source().is_none(),
            "an external-Indexer router must emit nothing rather than a structural zero",
        );
    }

    /// Every applied event is tallied by kind and medium, blocks included, so
    /// the scrape can show the tier stream the tree is consuming. An
    /// out-of-order batch is filtered before the tally and must not count.
    #[tokio::test]
    async fn pump_tallies_applied_events_by_medium() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tally, tx, pump) = (h.tally, h.tx, h.pump);

        let stored = |medium: Option<&str>, hashes: Vec<i64>| {
            KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: hashes,
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: medium.map(str::to_owned),
            })
        };
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 2,
            batch: batch(vec![
                stored(Some("GPU"), vec![10, 20, 30]),
                stored(Some("CPU_PINNED"), vec![10, 20, 30]),
                KvCacheEvent::BlockRemoved(BlockRemoved {
                    block_hashes: vec![30],
                    medium: Some("GPU".into()),
                }),
            ]),
        })
        .await
        .unwrap();
        // Out of order: filtered, must not be tallied.
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: batch(vec![stored(None, vec![99])]),
        })
        .await
        .unwrap();
        // A gap: seq 3 and 4 were dropped in transit. Counted, because a
        // tagged removal now clears only its own tier, so a lost batch can
        // strand a tier the tree will never clear on its own.
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 5,
            batch: batch(vec![KvCacheEvent::AllBlocksCleared]),
        })
        .await
        .unwrap();
        drop(tx);
        pump.await.unwrap();

        assert_eq!(tally.batches_lost(), 2, "seq 3 and 4 never arrived");
        let rows = tally.snapshot();
        let cell = |event: &str, medium: &str| {
            rows.iter()
                .find(|r| r.event == event && r.medium == medium)
                .cloned()
                .expect("cell rendered")
        };
        assert_eq!(cell("block_stored", "GPU").blocks, 3);
        assert_eq!(cell("block_stored", "CPU_PINNED").blocks, 3);
        assert_eq!(cell("block_removed", "GPU").events, 1);
        assert_eq!(
            cell("block_stored", "untagged").events,
            0,
            "the out-of-order batch was filtered before the tally",
        );
        assert_eq!(
            cell("all_blocks_cleared", "untagged").events,
            1,
            "a clear carries no medium and lands on the untagged row",
        );
    }

    /// A `WorkerEvent::Load` lands in the engine-load table (gauge, no
    /// cursor) keyed by the worker URL, and does not touch the tree.
    #[tokio::test]
    async fn pump_applies_load_to_engine_load_table() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tree, engine_reported_load, tx, pump) = (h.tree, h.engine_reported_load, h.tx, h.pump);

        tx.send(WorkerEvent::Load {
            worker: id.clone(),
            load: LoadStat {
                num_running_reqs: 8,
                num_waiting_reqs: 4,
                num_tokens: 0,
                max_total_num_tokens: 0,
                native_cache: None,
            },
        })
        .await
        .unwrap();
        drop(tx);
        pump.await.unwrap();

        let snapshot = engine_reported_load.capture_snapshot(Instant::now());
        let load = snapshot.fresh_load_for_url("http://w1").unwrap();
        assert_eq!(load.num_running_reqs + load.num_waiting_reqs, 12);
        // Load events must not pollute the cache tree.
        assert_eq!(tree.node_count(), 0);
    }

    /// Out-of-order seq is filtered: a batch with seq <= last_applied is
    /// dropped silently and does not mutate the tree.
    #[tokio::test]
    async fn pump_filters_out_of_order_seq() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tree, cursors, tx, pump) = (h.tree, h.cursors, h.tx, h.pump);

        // Apply seq=5 with block 10.
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 5,
            batch: batch(vec![KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![10],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: None,
            })]),
        })
        .await
        .unwrap();
        // Then a duplicate-style seq=3 that tries to remove block 10. Must
        // be dropped.
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 3,
            batch: batch(vec![KvCacheEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![10],
                medium: None,
            })]),
        })
        .await
        .unwrap();
        drop(tx);
        // Don't cancel — let rx.recv() return None naturally so any
        // queued events drain first. (The pump's `biased` select would
        // otherwise preempt unprocessed events on cancel.)
        pump.await.unwrap();

        let m = tree.match_prefix(None, &[10]);
        assert_eq!(
            m.matched_blocks, 1,
            "out-of-order remove must not undo the prior insert",
        );
        assert_eq!(cursors.lock().get(&id).copied(), Some(5));
    }

    /// AllBlocksCleared wipes the worker's tree state entirely.
    #[tokio::test]
    async fn pump_handles_all_blocks_cleared() {
        let id = worker_id("http://w1", 0);
        let h = spawn_pump(std::slice::from_ref(&id));
        let (tree, tx, pump) = (h.tree, h.tx, h.pump);

        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 1,
            batch: batch(vec![KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![1, 2],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: None,
            })]),
        })
        .await
        .unwrap();
        tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 2,
            batch: batch(vec![KvCacheEvent::AllBlocksCleared]),
        })
        .await
        .unwrap();
        drop(tx);
        // Don't cancel — let rx.recv() return None naturally so any
        // queued events drain first. (The pump's `biased` select would
        // otherwise preempt unprocessed events on cancel.)
        pump.await.unwrap();

        let m = tree.match_prefix(None, &[1, 2]);
        assert_eq!(
            m.matched_blocks, 0,
            "AllBlocksCleared must purge the worker"
        );
    }

    /// The pump drops events whose worker is not in `live_workers`. This
    /// is the safety net against the remove-then-pump race: an event
    /// queued before `remove_worker` clears the live set must not mutate
    /// the tree.
    #[tokio::test]
    async fn pump_drops_events_from_detached_workers() {
        let live_id = worker_id("http://live", 0);
        let dead_id = worker_id("http://dead", 0);
        let h = spawn_pump(std::slice::from_ref(&live_id));
        let (tree, tx, pump) = (h.tree, h.tx, h.pump);

        // Event from a worker that was never added (or was already
        // removed). Must be dropped.
        tx.send(WorkerEvent::Batch {
            worker: dead_id.clone(),
            seq: 1,
            batch: batch(vec![KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![42],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: None,
            })]),
        })
        .await
        .unwrap();
        // Sanity: a live event still applies.
        tx.send(WorkerEvent::Batch {
            worker: live_id.clone(),
            seq: 1,
            batch: batch(vec![KvCacheEvent::BlockStored(BlockStored {
                parent_block_hash: None,
                block_hashes: vec![99],
                token_ids: vec![],
                block_size: 64,
                lora_id: None,
                medium: None,
            })]),
        })
        .await
        .unwrap();
        drop(tx);
        // Don't cancel — let rx.recv() return None naturally so any
        // queued events drain first. (The pump's `biased` select would
        // otherwise preempt unprocessed events on cancel.)
        pump.await.unwrap();

        assert_eq!(tree.match_prefix(None, &[42]).matched_blocks, 0);
        assert_eq!(tree.match_prefix(None, &[99]).matched_blocks, 1);
    }

    /// `add_worker` must reject a worker whose `EventConfig.block_size`
    /// disagrees with the previously-established oracle value. The
    /// router cannot hash prompts simultaneously at two block sizes;
    /// silently accepting the mismatched worker would destroy
    /// cache-aware routing quality for every request.
    #[tokio::test]
    async fn add_worker_rejects_block_size_mismatch() {
        let index = KvEventIndex::new();
        // First worker establishes block_size=64 via the oracle.
        index.block_size_oracle().try_set(64).unwrap();

        let bad_cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30100,
            topic: String::new(),
            load_port_base: None,
            load_topic: None,
            block_size: 128,
            dp_size: 1,
            is_bigram: false,
        };
        index
            .add_worker("http://127.0.0.1:30100", Some(bad_cfg))
            .await;
        assert_eq!(
            index.known_worker_count(),
            0,
            "mismatched worker must not be registered"
        );
        index.shutdown().await;
    }

    #[tokio::test]
    async fn add_worker_seeds_oracle_with_first_block_size() {
        // Without any prior priming, the first worker through `add_worker`
        // should publish its `EventConfig.block_size` into the oracle so
        // subsequent matching workers reconcile and mismatched ones fail.
        let index = KvEventIndex::new();
        assert_eq!(index.block_size_oracle().get(), None);

        // A dp_size=0 cfg short-circuits before the subscriber spawn but
        // still runs through the block-size validation.
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30200,
            topic: String::new(),
            load_port_base: None,
            load_topic: None,
            block_size: 64,
            dp_size: 0,
            is_bigram: false,
        };
        index.add_worker("http://127.0.0.1:30200", Some(cfg)).await;
        assert_eq!(index.block_size_oracle().get(), Some(64));
        index.shutdown().await;
    }

    #[tokio::test]
    async fn add_worker_seeds_bigram_flag_from_event_config() {
        // The discovery->routing seam: add_worker must publish
        // EventConfig.is_bigram into the oracle (alongside block_size) so
        // select() picks the bigram hasher for EAGLE workers.
        let index = KvEventIndex::new();
        assert!(!index.block_size_oracle().is_bigram());
        // dp_size=0 short-circuits the subscriber spawn but still runs the seed.
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30300,
            topic: String::new(),
            load_port_base: None,
            load_topic: None,
            block_size: 64,
            dp_size: 0,
            is_bigram: true,
        };
        index.add_worker("http://127.0.0.1:30300", Some(cfg)).await;
        assert!(
            index.block_size_oracle().is_bigram(),
            "add_worker must seed the bigram flag from EventConfig"
        );
        index.shutdown().await;
    }

    #[tokio::test]
    async fn metadata_only_mode_keeps_the_load_subscriber() {
        let oracle = BlockSizeOracle::new();
        let index = KvEventIndex::new_metadata_only_with_http_and_oracle(
            reqwest::Client::new(),
            Arc::clone(&oracle),
        );
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 30400,
            topic: "kv-events".into(),
            load_port_base: Some(30410),
            load_topic: Some("load".into()),
            block_size: 64,
            dp_size: 2,
            is_bigram: true,
        };

        index.add_worker("http://127.0.0.1:30400", Some(cfg)).await;

        assert_eq!(oracle.get(), Some(64));
        assert!(oracle.is_bigram());
        assert_eq!(index.known_worker_count(), 1);
        assert_eq!(index.engine_reported_load().expected_count(), 1);
        index.shutdown().await;
    }

    /// `remove_worker` clears the worker's engine load and its expected mark,
    /// so a re-added worker does not inherit stale load. The worker advertises
    /// a load port (no publisher there; the subscriber just retries in the
    /// background and is cancelled on remove).
    #[tokio::test]
    async fn remove_worker_clears_engine_load() {
        let index = KvEventIndex::new();
        let url = "http://127.0.0.1:59123";
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 59123,
            topic: String::new(),
            load_port_base: Some(59223),
            load_topic: Some("load".into()),
            block_size: 64,
            dp_size: 1,
            is_bigram: false,
        };
        index.add_worker(url, Some(cfg)).await;
        assert_eq!(index.engine_reported_load().expected_count(), 1);

        let now = Instant::now();
        index.engine_reported_load().set(
            url,
            0,
            LoadStat {
                num_running_reqs: 3,
                num_waiting_reqs: 1,
                num_tokens: 0,
                max_total_num_tokens: 0,
                native_cache: None,
            },
            now,
        );
        assert!(index
            .engine_reported_load()
            .capture_snapshot(now)
            .fresh_load_for_url(url)
            .is_some());

        index.remove_worker(url).await;
        assert!(
            index
                .engine_reported_load()
                .capture_snapshot(Instant::now())
                .fresh_load_for_url(url)
                .is_none(),
            "remove_worker must clear engine load"
        );
        assert_eq!(index.engine_reported_load().expected_count(), 0);
        index.shutdown().await;
    }

    /// `remove_worker` returns only after every rank's tree state is cleared
    /// (the pump runs the teardown and signals `done`).
    #[tokio::test]
    async fn remove_worker_clears_every_rank_before_returning() {
        let index = KvEventIndex::new();
        let url = "http://127.0.0.1:59124";
        let cfg = EventConfig {
            host: "127.0.0.1".into(),
            port_base: 59124,
            topic: String::new(),
            load_port_base: None,
            load_topic: None,
            block_size: 64,
            dp_size: 2,
            is_bigram: false,
        };
        index.add_worker(url, Some(cfg)).await;

        // Stand in for events the pump already applied for both ranks.
        let tree = index.tree();
        let (r0, r1) = (worker_id(url, 0), worker_id(url, 1));
        tree.insert(&r0, None, &[10, 20]);
        tree.insert(&r1, None, &[30, 40]);
        assert!(tree.match_prefix(None, &[10, 20]).holds(&r0));
        assert!(tree.match_prefix(None, &[30, 40]).holds(&r1));

        // The pump is live, so the teardown below runs on it rather than on
        // the inline fallback.
        assert!(
            index.pump.lock().as_ref().is_some_and(|h| !h.is_finished()),
            "pump task must still be live, or the inline fallback does the teardown",
        );

        index.remove_worker(url).await;

        assert_eq!(
            tree.match_prefix(None, &[10, 20]).matched_blocks,
            0,
            "remove_worker must clear the removed worker's tree state",
        );
        assert_eq!(
            tree.match_prefix(None, &[30, 40]).matched_blocks,
            0,
            "every dp rank of the removed worker must be cleared",
        );
        assert_eq!(tree.node_count(), 0, "cleared chains must prune");
        index.shutdown().await;
    }

    /// A request's `max_age` is anchored at its arrival. An export sampled
    /// after the request arrived meets any `max_age`, including zero — that is
    /// what lets a waiter share the build it queued behind.
    #[test]
    fn max_age_is_measured_from_the_request_arrival() {
        let requested_at = Instant::now();
        let later = requested_at + Duration::from_secs(5);
        assert!(meets_max_age(later, requested_at, Duration::ZERO));
        assert!(meets_max_age(requested_at, requested_at, Duration::ZERO));

        let earlier = requested_at - Duration::from_millis(500);
        assert!(meets_max_age(earlier, requested_at, Duration::from_secs(1)));
        assert!(!meets_max_age(
            earlier,
            requested_at,
            Duration::from_millis(500)
        ));
        assert!(!meets_max_age(earlier, requested_at, Duration::ZERO));
    }

    /// A caller that gives up mid-build must not discard the build: the task
    /// finishes, fills the cache, and the next request is served from it.
    #[tokio::test]
    async fn an_abandoned_snapshot_request_still_fills_the_cache() {
        use futures::FutureExt;

        let index = KvEventIndex::new();
        // One poll takes the uncontended lock and hands the build to its task;
        // on the single-threaded test runtime that task cannot have run yet,
        // so the request is dropped with the build still pending.
        let abandoned = index.peer_snapshot_body(Duration::ZERO).now_or_never();
        assert!(
            abandoned.is_none(),
            "the request must be dropped mid-build, or this proves nothing",
        );
        let cache = index.snapshot_cache.lock().await;
        assert!(
            cache.as_ref().is_some_and(|c| !c.body.is_empty()),
            "the detached build must still cache its body",
        );
        drop(cache);
        index.shutdown().await;
    }
}
