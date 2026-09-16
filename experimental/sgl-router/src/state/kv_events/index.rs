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

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use flate2::write::GzEncoder;
use flate2::Compression;
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use super::block_size_oracle::BlockSizeOracle;
use super::bootstrap::{PeerSnapshot, WireWorker, SNAPSHOT_FORMAT};
use super::discovery::{fetch_event_config, EventConfig};
use super::subscriber::{KvEventSubscriberRegistry, SubKind, WorkerEvent};
use super::tally::{EventKind, EventTally};
use super::tree::{HashTree, KvWorkerId, SnapshotNode, Tiers};
use super::wire::KvCacheEvent;
use crate::state::load_monitor::engine_reported_load::EngineReportedLoadTable;

/// Channel buffer between the subscriber registry and the pump task.
///
/// Bounded so a misbehaving publisher cannot exhaust memory.  Realistic
/// per-worker event rates are < 1 kHz; a 1024-deep buffer absorbs a
/// half-second burst at 2 kHz before back-pressuring the SUB sockets.
const EVENT_CHANNEL_BUFFER: usize = 1024;

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
        Self::new_with_mode(http, block_size_oracle, true)
    }

    /// Discovers worker hash metadata only: seeds the shared [`BlockSizeOracle`]
    /// but neither subscribes to KV events nor maintains the local tree, because
    /// an external Indexer is the routing signal.
    pub fn new_metadata_only_with_http_and_oracle(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Arc<Self> {
        Self::new_with_mode(http, block_size_oracle, false)
    }

    fn new_with_mode(
        http: reqwest::Client,
        block_size_oracle: Arc<BlockSizeOracle>,
        maintain_tree: bool,
    ) -> Arc<Self> {
        let tree = Arc::new(HashTree::new());
        let (tx, rx) = mpsc::channel::<WorkerEvent>(EVENT_CHANNEL_BUFFER);
        let subscribers = Arc::new(KvEventSubscriberRegistry::new(tx.clone()));
        let load_subscribers = Arc::new(KvEventSubscriberRegistry::with_kind(tx, SubKind::Load));
        let engine_reported_load = EngineReportedLoadTable::new();
        let cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>> = Arc::new(Mutex::new(HashMap::new()));
        let live_workers: Arc<Mutex<HashSet<KvWorkerId>>> = Arc::new(Mutex::new(HashSet::new()));
        let pump_cancel = CancellationToken::new();
        let tally = Arc::new(EventTally::new());
        let pump = tokio::spawn(pump_loop(
            tree.clone(),
            engine_reported_load.clone(),
            cursors.clone(),
            live_workers.clone(),
            Arc::clone(&tally),
            pump_cancel.clone(),
            rx,
        ));
        Arc::new(Self {
            tree,
            maintain_tree,
            subscribers,
            load_subscribers,
            engine_reported_load,
            pump: Mutex::new(Some(pump)),
            pump_cancel,
            workers: Mutex::new(HashMap::new()),
            http,
            live_workers,
            cursors,
            tally,
            snapshot_cache: Arc::new(AsyncMutex::new(None)),
            block_size_oracle,
        })
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
            producer_ready: hash_config.is_some() && has_nodes,
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
        self.workers.lock().insert(
            worker_url.to_string(),
            WorkerEntry {
                dp_ranks: dp_ranks.clone(),
            },
        );
        if self.maintain_tree && !kv_dp_ranks.is_empty() {
            self.subscribers.add_worker(worker_url, &cfg).await;
        }
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
        // 3. Drop each rank's tree state and cursor, and the worker's engine
        //    load. Any event already in the mpsc buffer at this point will be
        //    filtered by the live-set check inside the pump.
        self.engine_reported_load.forget_worker(worker_url);
        let mut cursors = self.cursors.lock();
        for id in &ids {
            self.tree.clear_worker(id);
            cursors.remove(id);
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

/// Drain `WorkerEvent`s: apply KV `Batch`es to the tree and `Load` snapshots
/// to the engine-load table. Out-of-order (seq ≤ last_applied) and stale
/// (worker not in `live_workers`) KV batches are skipped; `Load` is a gauge
/// with no seq. `PublisherReset` events clear the cursor so a publisher
/// restarting from seq=1 (after sending END_SEQ) is not filtered.
async fn pump_loop(
    tree: Arc<HashTree>,
    engine_reported_load: Arc<EngineReportedLoadTable>,
    cursors: Arc<Mutex<HashMap<KvWorkerId, i64>>>,
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    tally: Arc<EventTally>,
    cancel: CancellationToken,
    mut rx: mpsc::Receiver<WorkerEvent>,
) {
    loop {
        let ev = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                info!("kv-events pump: shutdown requested; exiting");
                return;
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
                if cursors.lock().remove(&worker).is_some() {
                    info!(
                        worker = ?worker,
                        "kv-events pump: publisher reset; cursor cleared",
                    );
                }
            }
            WorkerEvent::Batch { worker, seq, batch } => {
                let prev = cursors.lock().get(&worker).copied();
                if let Some(p) = prev {
                    if seq <= p {
                        debug!(
                            worker = ?worker,
                            seq,
                            last_applied = p,
                            "kv-events pump: out-of-order batch; skipping",
                        );
                        continue;
                    }
                    // The publisher's seq is dense, so a jump is exactly the
                    // batches ZMQ dropped at its high-water mark. This became
                    // worth counting with tier-tagged removals: a removal now
                    // clears only its own tier, so losing the batch carrying a
                    // block's LAST removal leaves the worker owning it until
                    // the next AllBlocksCleared or teardown. The tree cannot
                    // see that happened — only the sequence can. The
                    // operator-visible signature is tree coverage above 1.
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
                    // The `medium` tag decides which tier a store lands on and
                    // which tier a removal clears, so a device eviction leaves
                    // a worker that still holds the block on host as an owner
                    // — see the tree's "Storage tiers" docs.
                    match event {
                        KvCacheEvent::BlockStored(b) => {
                            tally.record(
                                EventKind::BlockStored,
                                b.medium.as_deref(),
                                b.block_hashes.len(),
                            );
                            tree.insert_tiered(
                                &worker,
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
                                &worker,
                                &b.block_hashes,
                                Tiers::for_remove(b.medium.as_deref()),
                            );
                        }
                        KvCacheEvent::AllBlocksCleared => {
                            tally.record(EventKind::AllBlocksCleared, None, 0);
                            tree.clear_worker(&worker);
                        }
                    }
                }
                cursors.lock().insert(worker, seq);
            }
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
    }

    /// Build a tree + cursors + live-set wired through `pump_loop` with
    /// the given workers pre-marked live.
    fn spawn_pump(live: &[KvWorkerId]) -> PumpHarness {
        let tree = Arc::new(HashTree::new());
        let engine_reported_load = EngineReportedLoadTable::new();
        let cursors = Arc::new(Mutex::new(HashMap::new()));
        let live_set: Arc<Mutex<HashSet<KvWorkerId>>> =
            Arc::new(Mutex::new(live.iter().cloned().collect()));
        let cancel = CancellationToken::new();
        let tally = Arc::new(EventTally::new());
        let (tx, rx) = mpsc::channel(4);
        let pump = tokio::spawn(pump_loop(
            tree.clone(),
            engine_reported_load.clone(),
            cursors.clone(),
            live_set.clone(),
            Arc::clone(&tally),
            cancel.clone(),
            rx,
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
        }
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

    /// `remove_worker` is the tree's SECOND writer: it clears every rank's
    /// state from whatever task service discovery calls it on
    /// (`workers/manager.rs`) while the pump is live and writing the same
    /// tree. This pins the wiring that makes the collision
    /// `HashTree::concurrent_writers_never_orphan_a_chain` covers reachable
    /// at all — a refactor moving the clear onto the pump would retire it.
    #[tokio::test]
    async fn remove_worker_clears_the_tree_off_the_pump_task() {
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

        // Parked on its channel, not finished, so the clear below genuinely
        // runs on a different task than the one applying events.
        assert!(
            index.pump.lock().as_ref().is_some_and(|h| !h.is_finished()),
            "pump task must still be live, or this proves nothing about a second writer",
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
