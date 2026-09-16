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
    fetch_cursors, fetch_snapshot, BootstrapState, BootstrapTracker, FetchAnswer, PeerRegistry,
    PeerSnapshot, RankOutcome, SnapshotOutcome, SweepOutcome, VettedSnapshot, WireWorker,
    SNAPSHOT_FETCH_CONNECT_TIMEOUT, SNAPSHOT_FETCH_READ_TIMEOUT, SNAPSHOT_FORMAT,
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

/// Delay between peer-sweep attempts while no usable peer has been found.
///
/// Short relative to the bootstrap deadline that bounds the whole sweep, so a
/// peer becoming available is picked up promptly.
const PEER_RETRY_INTERVAL: Duration = Duration::from_millis(250);

/// Depth of the obligation queue feeding the coordinator, sized well past a
/// fleet's worker count. Overflow falls back as [`KvEventIndex::enqueue_bootstrap`]
/// describes.
const BOOTSTRAP_QUEUE_DEPTH: usize = 1024;

/// Passes to skip a peer after a retriable failure.
///
/// Unreachable, vet-failed and non-covering peers are retried after a few
/// passes: the peer may come back or discover our workers, and sitting out
/// spares a serving replica a multi-megabyte export every
/// [`PEER_RETRY_INTERVAL`].
const PEER_COOLDOWN_PASSES: u32 = 4;

/// Fetches the per-request timeout leaves room for within one deadline. The
/// sweep awaits candidates in turn, so one fetch must not consume the whole
/// deadline.
const SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE: u32 = 4;

/// Preferred floor for the per-fetch timeout; a snapshot body needs seconds to
/// transfer. Yields to `deadline / 2` and to the cap; see
/// [`snapshot_fetch_timeout`].
const SNAPSHOT_FETCH_TIMEOUT_FLOOR: Duration = Duration::from_secs(5);

/// Per-request timeout for a peer snapshot fetch, a strict fraction of the
/// configured bootstrap budget so several peers can be tried within it.
///
/// `cap` (from `--kv-bootstrap-fetch-timeout-cap-ms`, default
/// [`super::bootstrap::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP`]) bounds the
/// derivation, so a generous deadline (see `MAX_KV_BOOTSTRAP_TIMEOUT_MS`)
/// still bounds one progressing transfer; raise it when the snapshot body
/// outgrows the default.
///
/// The per-fetch bound is derived from the configured budget, not from the
/// time a given sweep has left, so a sweep running on a shrunken remainder is
/// bounded by its outer deadline rather than by this.
pub(crate) fn snapshot_fetch_timeout(deadline: Duration, cap: Duration) -> Duration {
    // A pre-settled (bootstrap-disabled) tracker reports a zero deadline. The
    // client is still built, and a zero reqwest timeout means "expire
    // immediately" rather than "no timeout", so fall back to the floor.
    if deadline.is_zero() {
        return SNAPSHOT_FETCH_TIMEOUT_FLOOR;
    }
    // Yielding to the cap keeps `clamp` from panicking on a below-floor cap.
    let floor = SNAPSHOT_FETCH_TIMEOUT_FLOOR.min(deadline / 2).min(cap);
    (deadline / SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE).clamp(floor, cap)
}

/// How long a grafted rank may wait for its own live stream to prove the splice
/// before the fleet is asked instead.
///
/// Nothing guarantees a first live batch ever comes: an idle rank would
/// otherwise serve grafted state forever with its continuity unproven, and a
/// `BlockRemoved` lost in the subscribe window would survive as a permanent
/// false cache hit. On expiry the rank is probed against the fleet's cursors,
/// not discarded; see the proof sweep in `pump_loop`.
const SPLICE_PROOF_TIMEOUT: Duration = Duration::from_secs(30);

/// Consecutive unanswerable probes after which an unproven graft is kept and
/// tallied [`RankOutcome::WarmUnwitnessed`].
///
/// Without a stop the pump would re-probe an idle rank forever whenever the
/// fleet is unreachable — a single-replica deployment being the obvious case —
/// and the rank's verdict would never resolve in the metrics. Keeping rather
/// than discarding follows the same reasoning as the probe itself.
const MAX_UNKNOWN_PROBES: u32 = 3;

/// How often the pump checks for splice proofs that never arrived. Coarse: the
/// deadline it enforces is [`SPLICE_PROOF_TIMEOUT`], not this.
const SPLICE_PROOF_SWEEP_INTERVAL: Duration = Duration::from_secs(5);

/// Upper bound on one probe pass across the fleet.
///
/// One sweep tick short of [`SPLICE_PROOF_TIMEOUT`], so a pass has always
/// reported before the sweep could launch the next one for the same ranks:
/// passes never overlap, and a slow or hung peer cannot stack up concurrent
/// passes that each re-ask every peer. A peer not reached inside the budget is
/// simply not a witness this pass, exactly as if it had been unreachable.
const SPLICE_PROBE_BUDGET: Duration =
    SPLICE_PROOF_TIMEOUT.saturating_sub(SPLICE_PROOF_SWEEP_INTERVAL);
const _: () = assert!(
    !SPLICE_PROBE_BUDGET.is_zero()
        && SPLICE_PROBE_BUDGET.as_nanos() + SPLICE_PROOF_SWEEP_INTERVAL.as_nanos()
            <= SPLICE_PROOF_TIMEOUT.as_nanos()
);

/// A grafted rank whose continuity with the live stream is not yet proven.
#[derive(Debug, Clone, Copy)]
struct PendingProof {
    /// Watermark the first arriving batch must not exceed by more than one.
    ///
    /// Also what a probe verdict must name to act on this entry: a gap retry
    /// re-grafts under the SAME incarnation, so the epoch alone cannot tell a
    /// verdict about the previous graft from one about this one.
    watermark: i64,
    /// When the graft happened, or when the last probe was launched.
    since: Instant,
    /// Consecutive probes that found no witness; see [`MAX_UNKNOWN_PROBES`].
    unknown_probes: u32,
}

impl PendingProof {
    fn new(watermark: i64, now: Instant) -> Self {
        Self {
            watermark,
            since: now,
            unknown_probes: 0,
        }
    }

    /// Whether the sweep should probe this rank now: once the graft, or the
    /// last probe, is older than `timeout`. A verdict that never lands (a
    /// panicked probe task, say) is therefore simply asked again.
    fn due_for_probe(&self, timeout: Duration) -> bool {
        self.since.elapsed() >= timeout
    }
}

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
    /// Result of asking the fleet whether a rank's publisher moved past the
    /// watermark of a snapshot whose splice was never proven locally.
    ///
    /// The probe runs off-pump because it does network I/O; the verdict comes
    /// back here so the tree write stays on the single writer.
    ///
    /// `epoch` and `watermark` name the graft the verdict is about; see
    /// [`PendingProof::watermark`].
    SpliceProbe {
        rank: KvWorkerId,
        epoch: u64,
        watermark: i64,
        verdict: SpliceVerdict,
    },
}

/// What the fleet says about a publisher's progress past an unproven watermark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpliceVerdict {
    /// Some peer has applied a sequence ABOVE our watermark. Sequence numbers
    /// come from the publisher, so that batch was emitted — and we never saw it,
    /// which is exactly the hole the splice check exists to catch.
    Advanced,
    /// A peer reported exactly the watermark and none is past it, so there is
    /// nothing we could have missed: the grafted state is continuous with a
    /// stream that has simply been silent.
    NoAdvance,
    /// Nobody answered, so the question stays open and the wait is re-armed.
    Unknown,
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

/// Obligations handed to the coordinator and the instant their ranks began
/// holding; a peer's export must be newer to splice.
struct ObligationBatch {
    obligations: Vec<(KvWorkerId, u64)>,
    /// Folded into [`PendingSweep::freshness_floor`], which the sweep asks with.
    holding_since: Instant,
    /// Whether this batch may ride a sweep already in flight.
    late_join: LateJoin,
}

/// What a batch accepts when it arrives while a sweep is already in flight.
///
/// The sweep asked for freshness on behalf of the ranks it started with, so a
/// batch that arrives afterwards may be delivered against an export predating
/// its own `holding_since` — which the pump then resolves [`RankOutcome::Gap`].
/// Whether that is acceptable depends on what the batch has left to spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LateJoin {
    /// Ride the in-flight sweep's snapshot regardless.
    ///
    /// Used by discovery, whose workers land during the first fetch; a rank
    /// that gaps this way still has its one retry.
    Permitted,
    /// Wait for a sweep that asks on this batch's behalf.
    ///
    /// Used by a gap retry, for which riding along is equivalent to dropping
    /// it: `gap_retried` caps it at one, and a snapshot taken before the rank
    /// resumed holding re-gaps by construction. Deferring costs one loop
    /// iteration, since the coordinator re-enters `take_pending` as soon as it
    /// has delivered.
    Refused,
}

/// Obligations taken off the queue for one sweep, with the freshness every one
/// of them requires.
struct PendingSweep {
    obligations: Vec<(KvWorkerId, u64)>,
    /// The newest `holding_since` among batches absorbed before the sweep: its
    /// one snapshot must postdate every rank in it. An `Instant` rather than a
    /// max age, so each retry's `elapsed()` asks for the same condition
    /// (`exported_at > floor`).
    freshness_floor: Instant,
    /// The newest `holding_since` of every batch folded in, including those
    /// admitted after the floor froze. What a batch built from this sweep's
    /// ranks must carry, since it may hold ranks that began holding after the
    /// floor.
    newest_holding: Instant,
    /// Batches that arrived too late for this sweep to have asked on their
    /// behalf and refused to ride along. Handed back for the next one.
    deferred: Vec<ObligationBatch>,
}

impl PendingSweep {
    /// Fold a batch in before the sweep starts, tightening the floor it will ask
    /// with.
    fn absorb(&mut self, batch: ObligationBatch) {
        self.freshness_floor = self.freshness_floor.max(batch.holding_since);
        self.fold(batch);
    }

    /// Add `batch`'s obligations without touching the floor.
    fn fold(&mut self, batch: ObligationBatch) {
        self.newest_holding = self.newest_holding.max(batch.holding_since);
        self.obligations.extend(batch.obligations);
    }

    /// Fold a batch in after the floor has been frozen — i.e. once the sweep is
    /// running and its request has already stated what it wants.
    ///
    /// Deliberately does NOT tighten the floor: the fetch has already gone out
    /// under the old one, so raising it here would describe a guarantee this
    /// sweep never asked for.
    fn admit_late(&mut self, batch: ObligationBatch) {
        if batch.late_join == LateJoin::Refused && batch.holding_since > self.freshness_floor {
            self.deferred.push(batch);
        } else {
            self.fold(batch);
        }
    }

    /// Keep only obligations a graft could still discharge, once each.
    ///
    /// A queued obligation can go stale before its sweep runs — the rank
    /// resolved, or its worker was removed and re-added — and would buy a
    /// fleet-wide fetch the pump then discards at its `Pending` gate. And the
    /// same rank can arrive twice (a coverage retry merged into a sweep that
    /// already holds it); a duplicate would be delivered alongside that rank's
    /// own retry and resolve it `Uncovered` before the retry lands.
    fn retain_graftable(&mut self, tracker: &BootstrapTracker) {
        // Keyed by epoch alone: the tracker mints epochs from one counter, and
        // the `epoch_of` check ties each surviving epoch to exactly one rank.
        let mut seen: HashSet<u64> = HashSet::with_capacity(self.obligations.len());
        self.obligations.retain(|(rank, epoch)| {
            tracker.epoch_of(rank) == Some(*epoch)
                && tracker.state_of(rank) == Some(BootstrapState::Pending)
                && seen.insert(*epoch)
        });
    }
}

impl From<ObligationBatch> for PendingSweep {
    fn from(batch: ObligationBatch) -> Self {
        Self {
            obligations: batch.obligations,
            freshness_floor: batch.holding_since,
            newest_holding: batch.holding_since,
            deferred: Vec::new(),
        }
    }
}

/// Run bootstrap sweeps one at a time, and let every rank queued while a sweep
/// is in flight share its snapshot.
///
/// A peer snapshot is fleet-wide — one body carries the whole worker table,
/// every cursor and the whole tree — so one fetch serves every pending rank,
/// where a sweep per discovered engine would re-download it once per engine.
/// The fetch itself is the batching window: it outlasts the watch event that
/// delivers a fleet, so ranks discovered meanwhile join its delivery at no
/// latency cost.
///
/// Sharing is safe because a graft is judged by sequence, not by clock: it
/// splices only when the rank's live stream resumes at `peer_cursor + 1`, and
/// each rank is judged on its own, so one whose publisher advanced resolves
/// [`RankOutcome::Gap`]. A rank the accepted peer does not cover gets one sweep
/// of its own before it resolves [`RankOutcome::Uncovered`].
///
/// Every fetch attempt states [`PendingSweep::freshness_floor`], which speaks
/// for the ranks the sweep started with. A rank merged in later may be
/// delivered against an older export; [`LateJoin`] says which batches accept
/// that.
async fn bootstrap_coordinator(
    mut rx: mpsc::Receiver<ObligationBatch>,
    index: std::sync::Weak<KvEventIndex>,
    cancel: CancellationToken,
) {
    // Obligations already given a second sweep for want of coverage. A rank no
    // peer covers must not buy a fresh fleet-wide fetch on every pass, so each
    // gets at most one retry. Keyed by incarnation, so a worker removed and re-added is
    // not denied its own, and pruned once that incarnation is gone.
    let mut requeued: HashSet<(KvWorkerId, u64)> = HashSet::new();
    loop {
        let Some(mut pending) = take_pending(&mut rx, &cancel).await else {
            return;
        };

        // Obligations already taken from the channel are dropped without a
        // `PumpControl` if this fails, which would normally strand their ranks in
        // `Pending`. Only reachable once the last `Arc<KvEventIndex>` is gone —
        // i.e. teardown, where the pump that would have received the message is
        // gone too and nothing is left to keep ready.
        let deps = {
            let Some(index) = index.upgrade() else { return };
            index.bootstrap_deps()
        };
        requeued.retain(|(rank, epoch)| deps.bootstrap.epoch_of(rank) == Some(*epoch));
        pending.retain_graftable(&deps.bootstrap);
        if pending.obligations.is_empty() {
            continue;
        }
        let deadline = deps.deadline();
        let ranks: Vec<KvWorkerId> = pending.obligations.iter().map(|(r, _)| r.clone()).collect();

        // ONE sweep, awaited here rather than spawned — that is what makes this
        // single-flight. Obligations discovered while it runs pile up in the
        // channel and are merged below, so they ride this same snapshot.
        let started = Instant::now();
        info!(
            ranks = ranks.len(),
            peers = deps.peers.len(),
            deadline_ms = deadline.as_millis(),
            "kv-bootstrap: sweeping sibling replicas for a tree snapshot",
        );
        // Raced against shutdown, which cancels the pump through the same token:
        // a sweep finishing after it has nobody to deliver to, and would keep
        // pulling multi-megabyte bodies from siblings for up to the deadline.
        let result = tokio::select! {
            _ = cancel.cancelled() => return,
            result = sweep_until_deadline(&deps, &ranks, deadline, pending.freshness_floor) => result,
        };
        drop(ranks);
        let joined = drain_ready(&mut rx, &mut pending);
        if joined > 0 || !pending.deferred.is_empty() {
            debug!(
                joined,
                deferred = pending.deferred.len(),
                total = pending.obligations.len(),
                sweep_ms = started.elapsed().as_millis(),
                "kv-bootstrap: ranks arriving during the sweep joined its snapshot, \
                 minus any that need a fresher one",
            );
        }
        // Late arrivals can repeat a rank already here, and anything can have
        // resolved or been removed while the sweep ran.
        pending.retain_graftable(&deps.bootstrap);

        // The sweep's coverage check (`covers_any`) only spoke for the ranks it
        // was given, so a rank merged afterwards may be absent from the accepted
        // peer's tree — the peer can itself be partially discovered and know one
        // of our workers but not another. Delivering such a rank into this
        // snapshot resolves it `Uncovered`, which is TERMINAL, where a sweep that
        // asked on its behalf would have kept looking for a peer that did cover
        // it. Give it that look instead.
        if let SweepResult::Found(ref vetted) = result {
            let covered = vetted.covered_ranks();
            let (deliver, retry): (Vec<_>, Vec<_>) = std::mem::take(&mut pending.obligations)
                .into_iter()
                .partition(|ob| covered.contains(&ob.0) || !requeued.insert(ob.clone()));
            pending.obligations = deliver;
            if !retry.is_empty() {
                // The newest instant anything here began holding, not the frozen
                // floor: a rank merged in after the sweep started began holding
                // after the floor, and asking only for the floor would let its
                // retry accept an export that predates it. Still not NOW, which
                // would force the peer into rebuilds these ranks do not need.
                pending.deferred.push(ObligationBatch {
                    obligations: retry,
                    holding_since: pending.newest_holding,
                    late_join: LateJoin::Permitted,
                });
            }
        }

        // Re-queue via the index rather than a sender this task owns: a
        // long-lived clone here would hold the channel open forever and kill
        // the all-senders-dropped exit.
        let live = index.upgrade();
        for batch in std::mem::take(&mut pending.deferred) {
            match &live {
                // Never folded into this delivery: that would spend the
                // freshness these ranks asked for.
                Some(idx) => idx.enqueue_bootstrap(batch),
                // Teardown: nothing is left to run another sweep, so deliver them
                // here rather than stranding them.
                None => pending.obligations.extend(batch.obligations),
            }
        }
        // Not held across the delivery, which would keep the index alive.
        drop(live);

        deliver_bootstrap(&deps, pending.obligations, result, deadline).await;
    }
}

/// Block for the first obligation, then take everything already queued behind it
/// without waiting.
///
/// `None` means stop — cancelled, or every sender dropped with nothing pending.
async fn take_pending(
    rx: &mut mpsc::Receiver<ObligationBatch>,
    cancel: &CancellationToken,
) -> Option<PendingSweep> {
    // Block until there is something to do, so an idle fleet costs nothing.
    let first = tokio::select! {
        _ = cancel.cancelled() => return None,
        first = rx.recv() => first?,
    };
    let mut pending = PendingSweep::from(first);
    // Pre-sweep, so these tighten the floor the sweep will ask with rather than
    // arriving after it has already asked.
    while let Ok(more) = rx.try_recv() {
        pending.absorb(more);
    }
    Some(pending)
}

/// Move every immediately-available obligation into `into`, returning how many
/// ranks were added. Never waits.
///
/// Called once the sweep has run, so batches are admitted under the frozen floor
/// — see [`PendingSweep::admit_late`]. Ranks it defers are not counted as joined.
fn drain_ready(rx: &mut mpsc::Receiver<ObligationBatch>, into: &mut PendingSweep) -> usize {
    let before = into.obligations.len();
    while let Ok(more) = rx.try_recv() {
        into.admit_late(more);
    }
    into.obligations.len() - before
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
    /// Client for snapshot fetches. Not `http`: its 2s total timeout suits
    /// `/server_info` but cannot fit a multi-megabyte body, and every large
    /// snapshot would be booked `unreachable`.
    snapshot_http: reqwest::Client,
    /// Obligations waiting for the coordinator to fold them into the sweep that
    /// is in flight, or to start one. See [`bootstrap_coordinator`].
    bootstrap_tx: mpsc::Sender<ObligationBatch>,
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
        // Connect and read timeouts cut a gone or stalled peer; the total
        // bounds a progressing transfer to a fraction of the deadline so the
        // sweep can reach another candidate (see `snapshot_fetch_timeout`).
        let per_fetch = snapshot_fetch_timeout(bootstrap.timeout(), bootstrap.fetch_cap());
        // A short `--kv-bootstrap-timeout-ms` can derive a per-fetch bound
        // below the floor that the cap cannot lift; warn once so the operator
        // raises the deadline.
        if bootstrap.enabled() && per_fetch < SNAPSHOT_FETCH_TIMEOUT_FLOOR {
            warn!(
                per_fetch_ms = per_fetch.as_millis(),
                bootstrap_timeout_ms = bootstrap.timeout().as_millis(),
                fetch_cap_ms = bootstrap.fetch_cap().as_millis(),
                floor_ms = SNAPSHOT_FETCH_TIMEOUT_FLOOR.as_millis(),
                suggested_bootstrap_timeout_ms = (SNAPSHOT_FETCH_TIMEOUT_FLOOR
                    * SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE)
                    .as_millis(),
                "kv-bootstrap: the per-fetch timeout derived from --kv-bootstrap-timeout-ms \
                 is below the floor a multi-megabyte snapshot needs, so peers will be \
                 booked unreachable and every rank will boot cold; raise \
                 --kv-bootstrap-timeout-ms (the fetch cap cannot lift this on its own)",
            );
        }
        let snapshot_http = match reqwest::Client::builder()
            .connect_timeout(SNAPSHOT_FETCH_CONNECT_TIMEOUT)
            .read_timeout(SNAPSHOT_FETCH_READ_TIMEOUT)
            .timeout(per_fetch)
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                // Not silent: the fallback's total timeout is sized for
                // `/server_info`, so every large snapshot would then time out
                // and be booked `unreachable` with nothing pointing here.
                warn!(
                    error = %e,
                    "kv-bootstrap: snapshot client failed to build; falling back to the \
                     introspection client, whose timeout cannot fit a large snapshot",
                );
                http.clone()
            }
        };
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
        let (bootstrap_tx, bootstrap_rx) = mpsc::channel(BOOTSTRAP_QUEUE_DEPTH);
        let tally = Arc::new(EventTally::new());
        let pump = tokio::spawn(pump_loop(
            PumpDeps {
                tally: Arc::clone(&tally),
                tree: tree.clone(),
                engine_reported_load: engine_reported_load.clone(),
                cursors: cursors.clone(),
                live_workers: live_workers.clone(),
                bootstrap: Arc::clone(&bootstrap),
                peers: Arc::clone(&peers),
                snapshot_http: snapshot_http.clone(),
                bootstrap_tx: bootstrap_tx.clone(),
                ctrl_tx: ctrl_tx.downgrade(),
            },
            pump_cancel.clone(),
            rx,
            ctrl_rx,
        ));
        let index = Arc::new(Self {
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
            snapshot_http,
            bootstrap_tx,
            snapshot_cache: Arc::new(AsyncMutex::new(None)),
            block_size_oracle,
        });
        // Same gate as registration, so a coordinator exists exactly when
        // obligations can be produced.
        if index.peer_bootstrap_enabled() {
            tokio::spawn(bootstrap_coordinator(
                bootstrap_rx,
                Arc::downgrade(&index),
                pump_cancel,
            ));
        }
        index
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
        let bootstrap_obligations = self.register_for_bootstrap(&bootstrap_ranks);
        self.workers.lock().insert(
            worker_url.to_string(),
            WorkerEntry {
                dp_ranks: dp_ranks.clone(),
            },
        );
        if self.maintain_tree && !kv_dp_ranks.is_empty() {
            self.subscribers.add_worker(worker_url, &cfg).await;
        }
        if !bootstrap_obligations.is_empty() {
            // Stamped HERE, after `subscribers.add_worker` — not at `register`.
            // A peer's export has to beat the subscription, and anything earlier
            // would let the sweep accept a snapshot taken during the subscribe
            // window, which is precisely the hole the watermark check would then
            // reject as `Gap`. `subscribers.add_worker` only spawns the SUB
            // tasks, though: the connect completes asynchronously, so an export
            // taken between this stamp and the connect can still gap, and the
            // splice check is what catches it.
            let batch = ObligationBatch {
                obligations: bootstrap_obligations,
                holding_since: Instant::now(),
                late_join: LateJoin::Permitted,
            };
            // The stamp travels with the batch, so whichever sweep picks it up
            // asks for an export newer than it.
            self.enqueue_bootstrap(batch);
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

    /// Whether this worker's ranks enter the bootstrap state machine. Keyed on
    /// configured, not settled, so a late worker still gets an obligation; and
    /// independent of visible peers, because registering arms the deadline.
    fn peer_bootstrap_enabled(&self) -> bool {
        self.bootstrap.enabled()
    }

    /// Register the ranks a sweep should run for, returning their obligations.
    ///
    /// Only ranks the tracker does not already hold yield one:
    /// `reconcile_unresolved_workers` re-calls `add_worker` for a worker it
    /// already knows, and a rank that is `Pending` already has a sweep while a
    /// terminal one has nothing left to fetch. A rank `remove_worker` forgot
    /// registers as a new incarnation.
    fn register_for_bootstrap(&self, ranks: &[KvWorkerId]) -> Vec<(KvWorkerId, u64)> {
        if !self.peer_bootstrap_enabled() {
            return Vec::new();
        }
        self.bootstrap.register(ranks)
    }

    /// Hand `batch` to the coordinator, or sweep it on its own when the queue is
    /// full or the coordinator is gone.
    ///
    /// Never drops it: an obligation nothing owns leaves its ranks `Pending`,
    /// which holds `/readyz` and holds their batches until the pump's per-rank
    /// cap overflows. Sweeping un-batched costs a fetch the coordinator would
    /// have shared, and keeps the freshness the batch asked for.
    fn enqueue_bootstrap(&self, batch: ObligationBatch) {
        if let Err(e) = self.bootstrap_tx.try_send(batch) {
            warn!("kv-bootstrap: batch queue unavailable ({e}); sweeping un-batched");
            self.spawn_bootstrap(e.into_inner());
        }
    }

    /// Fetch a snapshot for one batch of obligations from the first peer, in
    /// shuffled order, whose snapshot vets and covers at least one of its ranks,
    /// and hand the result to the pump.
    ///
    /// Runs detached: `/readyz` is gated by the tracker, not by awaiting this,
    /// so a slow peer delays readiness only up to the bootstrap deadline. Every
    /// exit path sends exactly one [`PumpControl`] message, which is what
    /// guarantees the held-back batches are eventually released.
    fn spawn_bootstrap(&self, batch: ObligationBatch) {
        let deps = self.bootstrap_deps();
        tokio::spawn(async move {
            let ObligationBatch {
                obligations,
                holding_since,
                late_join: _,
            } = batch;
            let ranks: Vec<KvWorkerId> = obligations.iter().map(|(r, _)| r.clone()).collect();
            let deadline = deps.deadline();
            let result = sweep_until_deadline(&deps, &ranks, deadline, holding_since).await;
            deliver_bootstrap(&deps, obligations, result, deadline).await;
        });
    }

    /// Clone the handles a sweep needs, so it can run detached — or be awaited
    /// by the coordinator — without borrowing `self`.
    fn bootstrap_deps(&self) -> BootstrapDeps {
        BootstrapDeps {
            http: self.snapshot_http.clone(),
            peers: Arc::clone(&self.peers),
            bootstrap: Arc::clone(&self.bootstrap),
            live_workers: Arc::clone(&self.live_workers),
            oracle: Arc::clone(&self.block_size_oracle),
            ctrl_tx: self.ctrl_tx.clone(),
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

/// The handles a bounded peer sweep needs, cloned out of [`KvEventIndex`] so the
/// sweep can run detached — or be awaited by the coordinator — without
/// borrowing `self`.
struct BootstrapDeps {
    http: reqwest::Client,
    peers: Arc<PeerRegistry>,
    bootstrap: Arc<BootstrapTracker>,
    live_workers: Arc<Mutex<HashSet<KvWorkerId>>>,
    oracle: Arc<BlockSizeOracle>,
    ctrl_tx: mpsc::Sender<PumpControl>,
}

impl BootstrapDeps {
    /// Budget for one sweep. Before readiness settles, the remainder of the
    /// tracker's single `/readyz` window; once settled, a full `timeout()` for
    /// a late-discovered worker (`time_remaining` saturates at zero).
    fn deadline(&self) -> Duration {
        if self.bootstrap.settled() {
            return self.bootstrap.timeout();
        }
        self.bootstrap
            .time_remaining()
            .unwrap_or(self.bootstrap.timeout())
    }
}

/// Outcome of one bounded peer sweep.
enum SweepResult {
    Found(VettedSnapshot),
    /// Discovery confirmed there are no siblings, so waiting cannot help.
    NoPeers,
    /// Every candidate's latest word was cold-or-terminal: an empty tree, or a
    /// permanent incompatibility. Waiting out the deadline cannot help —
    /// anything these peers learn later arrives over this replica's own event
    /// subscriptions anyway.
    FleetCold {
        /// Size of the candidate set the verdict was proven over — not the
        /// live registry, which may have changed since.
        peers_tried: usize,
    },
    TimedOut {
        /// Registry size at the deadline.
        peers_tried: usize,
        last_reason: Option<String>,
    },
}

impl SweepResult {
    /// The metric-facing verdict, mirroring `VetError::outcome`: the closed
    /// label set lives in bootstrap.rs so no call site can mint new values.
    fn outcome(&self) -> SweepOutcome {
        match self {
            Self::Found(_) => SweepOutcome::Found,
            Self::NoPeers => SweepOutcome::NoPeers,
            Self::FleetCold { .. } => SweepOutcome::FleetCold,
            Self::TimedOut { .. } => SweepOutcome::TimedOut,
        }
    }

    /// Size of the candidate set the verdict was proven over. `Found` reports
    /// 1 — the peer that answered — so a success is never mistaken for a
    /// verdict reached over nobody.
    fn peers_tried(&self) -> usize {
        match self {
            Self::Found(_) => 1,
            Self::NoPeers => 0,
            Self::FleetCold { peers_tried } | Self::TimedOut { peers_tried, .. } => *peers_tried,
        }
    }
}

/// Sweep peers until one yields a usable snapshot, every sibling proves it
/// has nothing to give, discovery proves there are none, or the budget runs
/// out.
///
/// Retries every `PEER_RETRY_INTERVAL` because the peer watch can deliver its
/// first list after worker discovery. Settles early when every candidate
/// answers cold or incompatible. `freshness_floor` is re-derived into a max
/// age per attempt.
async fn sweep_until_deadline(
    deps: &BootstrapDeps,
    ranks: &[KvWorkerId],
    deadline: Duration,
    freshness_floor: Instant,
) -> SweepResult {
    let ctx = SweepCtx {
        http: &deps.http,
        peers: &deps.peers,
        bootstrap: &deps.bootstrap,
        live_workers: &deps.live_workers,
        oracle: &deps.oracle,
        freshness_floor,
    };
    // Shared so the terminal log can name the last concrete reason rather than
    // only "no usable snapshot".
    let last_reason: Mutex<Option<String>> = Mutex::new(None);
    let attempt = async {
        let mut state = SweepState::new();
        loop {
            match sweep_peers(&ctx, ranks, &mut state, &last_reason).await {
                SweepPass::Found(vetted) => return Some(SweepResult::Found(vetted)),
                SweepPass::FleetCold { peers_tried } => {
                    return Some(SweepResult::FleetCold { peers_tried })
                }
                SweepPass::KeepLooking => {}
            }
            if deps.peers.known_to_have_no_peers() {
                debug!("kv-bootstrap: discovery confirmed no sibling replicas");
                return None;
            }
            tokio::time::sleep(PEER_RETRY_INTERVAL).await;
        }
    };

    // The deadline bounds the whole sweep, not each request, so a fleet of slow
    // peers cannot outlast the readiness gate.
    match tokio::time::timeout(deadline, attempt).await {
        Ok(Some(result)) => result,
        Ok(None) => SweepResult::NoPeers,
        Err(_) => SweepResult::TimedOut {
            peers_tried: deps.peers.len(),
            last_reason: last_reason.lock().clone(),
        },
    }
}

/// Turn a sweep result into the single [`PumpControl`] message its obligations
/// are owed. Every exit path sends exactly one, which is what releases the ranks
/// from `Pending`.
async fn deliver_bootstrap(
    deps: &BootstrapDeps,
    obligations: Vec<(KvWorkerId, u64)>,
    result: SweepResult,
    deadline: Duration,
) {
    let n = obligations.len();
    deps.bootstrap
        .record_sweep_result(result.outcome(), result.peers_tried());
    match &result {
        SweepResult::Found(_) => {}
        SweepResult::NoPeers => info!(
            ranks = n,
            "kv-bootstrap: no sibling replicas to bootstrap from; ranks will run cold",
        ),
        SweepResult::FleetCold { peers_tried } => info!(
            ranks = n,
            peers_tried,
            "kv-bootstrap: every sibling replica answered with an empty or \
             incompatible tree; settling cold without waiting out the deadline",
        ),
        SweepResult::TimedOut {
            peers_tried,
            last_reason,
        } => warn!(
            ranks = n,
            timeout_ms = deadline.as_millis(),
            peers_tried,
            last_reason = last_reason.as_deref().unwrap_or("none recorded"),
            "kv-bootstrap: no peer supplied a usable snapshot within the deadline; \
             ranks will run cold",
        ),
    }
    let msg = match result {
        SweepResult::Found(vetted) => PumpControl::ApplySnapshot {
            obligations,
            vetted: Box::new(vetted),
        },
        _ => PumpControl::AbandonBootstrap { obligations },
    };
    if deps.ctrl_tx.send(msg).await.is_err() {
        warn!("kv-bootstrap: pump is gone; bootstrap result discarded");
    }
}

struct SweepCtx<'a> {
    http: &'a reqwest::Client,
    peers: &'a PeerRegistry,
    bootstrap: &'a BootstrapTracker,
    live_workers: &'a Mutex<HashSet<KvWorkerId>>,
    oracle: &'a BlockSizeOracle,
    /// See [`sweep_until_deadline`]. Held as the instant, converted to an age at
    /// each fetch.
    freshness_floor: Instant,
}

/// Verdict of one pass over the candidate peers.
enum SweepPass {
    /// A peer's snapshot vetted and covers ranks we are bootstrapping.
    Found(VettedSnapshot),
    /// Nothing usable this pass, but at least one peer might have state later
    /// (unanswered, still bootstrapping, or warm-but-not-covering).
    KeepLooking,
    /// Every candidate's latest word was cold-or-terminal. Carries the size of
    /// the candidate set the verdict was proven over; see
    /// [`SweepResult::FleetCold`].
    FleetCold { peers_tried: usize },
}

/// Per-sweep peer state carried across passes.
struct SweepState {
    /// Peers whose snapshot is permanently incompatible (format, block size).
    /// A stable property of the peer for the life of the process, so they are
    /// never re-fetched.
    permanently_rejected: HashSet<String>,
    /// Retriable-failure sit-out, in remaining passes.
    cooldown: HashMap<String, u32>,
    /// Peers whose latest answer was an empty tree
    /// ([`PeerSnapshot::holds_no_state`], NOT the stricter
    /// [`PeerSnapshot::is_cold`] vetting uses — a peer mid-bootstrap that
    /// already holds nodes is not done, but it plainly HAS state). Carried
    /// across passes so a peer in cooldown keeps its last classification;
    /// dropped on any warmer or unknown answer, since the fleet proving cold
    /// is only meaningful when EVERY peer's latest word is "I have nothing".
    cold_witnessed: HashSet<String>,
}

impl SweepState {
    fn new() -> Self {
        Self {
            permanently_rejected: HashSet::new(),
            cooldown: HashMap::new(),
            cold_witnessed: HashSet::new(),
        }
    }

    /// A peer that did not answer has unknown warmth: it cannot count toward
    /// the all-cold verdict, and it sits out a few passes — the failures that
    /// land here are mostly stable for the life of the process (a transport it
    /// cannot complete, a body that will not inflate or parse), so re-fetching
    /// a multi-megabyte body from it on every pass is pure load on a replica
    /// that is itself serving traffic.
    fn note_unreachable(&mut self, peer: &str) {
        self.cold_witnessed.remove(peer);
        self.cooldown.insert(peer.to_string(), PEER_COOLDOWN_PASSES);
    }

    /// Record the temperature of an answered snapshot: an empty tree is a
    /// cold witness; anything with content — usable, still bootstrapping, or
    /// not covering us — means the fleet holds state worth waiting for.
    fn note_answer(&mut self, peer: &str, snap: &PeerSnapshot) {
        if snap.holds_no_state() {
            self.cold_witnessed.insert(peer.to_string());
        } else {
            self.cold_witnessed.remove(peer);
        }
    }

    /// Permanent rejection is terminal on its own — a peer whose state we can
    /// never consume has nothing to give us, whatever it holds.
    fn note_permanent_reject(&mut self, peer: &str) {
        self.permanently_rejected.insert(peer.to_string());
    }

    /// Sit out a few passes without touching the cold witness; contrast
    /// `note_unreachable`.
    fn note_sit_out(&mut self, peer: &str) {
        self.cooldown.insert(peer.to_string(), PEER_COOLDOWN_PASSES);
    }

    /// True while `peer` sits out a retriable failure, decaying one pass.
    fn cooling(&mut self, peer: &str) -> bool {
        match self.cooldown.get_mut(peer) {
            Some(remaining) if *remaining > 0 => {
                *remaining -= 1;
                true
            }
            _ => false,
        }
    }

    /// The all-cold verdict: every candidate's latest classification says it
    /// has nothing to give. Peers skipped by cooldown count by their last
    /// classification; a candidate that has never answered (new to the set,
    /// or only ever unreachable) keeps the sweep waiting.
    ///
    /// An empty candidate set is NOT cold: `all()` on an empty iterator is
    /// vacuously true, and the peer watch regularly delivers its first list
    /// after worker discovery completes. An empty set means "no information"
    /// — handled by the `known_to_have_no_peers` path — not "everyone proved
    /// empty".
    fn fleet_is_cold(&self, candidates: &[String]) -> bool {
        !candidates.is_empty()
            && candidates
                .iter()
                .all(|p| self.permanently_rejected.contains(p) || self.cold_witnessed.contains(p))
    }
}

/// One pass over the candidates, returning the first snapshot that vets and
/// covers a rank.
async fn sweep_peers(
    ctx: &SweepCtx<'_>,
    ranks: &[KvWorkerId],
    state: &mut SweepState,
    last_reason: &Mutex<Option<String>>,
) -> SweepPass {
    let SweepCtx {
        http,
        peers,
        bootstrap,
        live_workers,
        oracle,
        freshness_floor,
    } = ctx;
    // Both halves or neither: acting on a block size whose companion hashing
    // mode is not yet published would vet a peer's tree against an identity
    // this replica has not established. See `BlockSizeOracle::hash_config`.
    // Only the size is vetted on — hashing mode deliberately is not, see
    // `VettedSnapshot::from_wire`.
    let Some((local_block_size, local_bigram)) = oracle.hash_config() else {
        return SweepPass::KeepLooking;
    };
    // A fetch that yielded nothing: name it for the terminal log and tally it.
    // The `last_reason` guard drops at the end of its statement.
    let miss = |peer: &str, outcome: SnapshotOutcome, detail: &str| {
        *last_reason.lock() = Some(format!("{peer}: {detail}"));
        bootstrap.record_peer_outcome(outcome, peer, Some(detail));
    };
    let candidates = peers.candidates();
    for peer in &candidates {
        if state.permanently_rejected.contains(peer) {
            continue;
        }
        if state.cooling(peer) {
            continue;
        }
        // Ask for an export that beats the floor. Derived per attempt, not once:
        // the condition is "newer than the floor", and only the age it
        // corresponds to moves as the sweep retries.
        let max_age = freshness_floor.elapsed();
        let fetched = match fetch_snapshot(http, peer, Some(max_age)).await {
            Ok(FetchAnswer::Body(s)) => Ok(s),
            // Reachable but no usable body — the status names which kind of
            // wrong: 404 is an older router image that does not serve the
            // route, 5xx is a sick sibling. Both retriable.
            Ok(FetchAnswer::NoBody(status)) => Err(format!("answered HTTP {status}")),
            // `{e:#}` for the anyhow chain: the bare Display prints only the
            // outermost message, dropping the reqwest/io cause that names what
            // actually went wrong.
            Err(e) => Err(format!("{e:#}")),
        };
        let snap = match fetched {
            Ok(snap) => snap,
            Err(detail) => {
                state.note_unreachable(peer);
                miss(peer, SnapshotOutcome::Unreachable, &detail);
                continue;
            }
        };
        // Classify before vetting: an empty tree is a cold witness even when the
        // snapshot is otherwise well formed.
        state.note_answer(peer, &snap);
        let peer_bigram = snap.is_bigram;
        // Snapshot the live set at vet time so a peer cannot introduce a worker
        // this replica has not discovered.
        let live = live_workers.lock().clone();
        match VettedSnapshot::from_wire(snap, &live, Some(local_block_size)) {
            Ok(vetted) => {
                // Vetting only proves the snapshot is well formed and hash-
                // comparable. It can still know nothing about the ranks we are
                // bootstrapping, in which case accepting it would end the sweep
                // and leave those ranks cold.
                if !vetted.covers_any(ranks) {
                    debug!(
                        peer = %peer,
                        nodes = vetted.node_count(),
                        "kv-bootstrap: peer has no state for the ranks being bootstrapped; \
                         continuing to look",
                    );
                    // Usually a warm peer that has not yet discovered our worker.
                    // Fixed text so the per-peer log limiter dedups repeats.
                    state.note_sit_out(peer);
                    miss(
                        peer,
                        SnapshotOutcome::ColdPeer,
                        "snapshot tracks none of the ranks being bootstrapped",
                    );
                    continue;
                }
                // How much of the copy can steer a selection: carried nodes
                // answer a query, structure nodes are match paths only. See
                // `VettedSnapshot::carrier_counts`.
                let (carried, structure) = vetted.carrier_counts();
                // The producer's hashing mode is logged rather than vetted
                // (see `VettedSnapshot::from_wire`), but a disagreement means
                // the fleet is mid-rollout across a spec-config change and the
                // grafted blocks may not match — worth naming when it happens.
                if peer_bigram != local_bigram {
                    info!(
                        peer = %peer,
                        peer_bigram,
                        local_bigram,
                        "kv-bootstrap: peer reports a different fleet hashing mode; \
                         grafting anyway, since the surviving carriers are workers \
                         this replica discovered itself",
                    );
                }
                info!(
                    peer = %peer,
                    nodes = vetted.node_count(),
                    carried_nodes = carried,
                    structure_nodes = structure,
                    workers = vetted.worker_count(),
                    dropped_workers = vetted.dropped_workers(),
                    "kv-bootstrap: snapshot accepted; handing to pump",
                );
                bootstrap.record_peer_outcome(SnapshotOutcome::Accepted, peer, None);
                return SweepPass::Found(vetted);
            }
            Err(e) => {
                if e.outcome() == SnapshotOutcome::Rejected {
                    // Loud: a fleet-wide block-size or format disagreement means
                    // NO replica can ever bootstrap, and at debug level the only
                    // symptom is a generic deadline warning.
                    warn!(
                        peer = %peer,
                        error = %e,
                        "kv-bootstrap: peer snapshot is permanently incompatible; \
                         not retrying this peer",
                    );
                    state.note_permanent_reject(peer);
                }
                state.note_sit_out(peer);
                miss(peer, e.outcome(), &e.to_string());
            }
        }
    }
    if !state.fleet_is_cold(&candidates) {
        return SweepPass::KeepLooking;
    }
    // Discard the cold verdict if candidate membership changed mid-pass
    // (compared as sets; a same-length swap counts): the newcomer was never
    // consulted.
    if peers.candidates().into_iter().collect::<HashSet<_>>()
        != candidates.iter().cloned().collect::<HashSet<_>>()
    {
        // Not silent: this is how a flapping EndpointSlice turns a cold
        // fleet's quick settle into a full-deadline wait.
        info!("kv-bootstrap: candidate set changed mid-pass; discarding the cold-fleet verdict");
        return SweepPass::KeepLooking;
    }
    SweepPass::FleetCold {
        peers_tried: candidates.len(),
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
    /// Peer set and client for the splice probe; see `spawn_splice_probe`. The
    /// pump does not fetch snapshots for bootstrap itself — only this one
    /// question, about state it already grafted.
    peers: Arc<PeerRegistry>,
    snapshot_http: reqwest::Client,
    /// Obligation queue, so a gap-discarded rank can be handed back for another
    /// sweep instead of staying cold with budget unspent.
    bootstrap_tx: mpsc::Sender<ObligationBatch>,
    /// Loopback into this pump's own control channel, so a probe answer arrives
    /// on the single writer like every other tree mutation.
    ///
    /// Weak so the pump does not keep its own control channel open: the channel
    /// still closes when every external sender is gone, which is what the
    /// `ctrl_open` latch in `pump_loop` exists to observe. A probe upgrades it
    /// for the duration of its pass.
    ctrl_tx: mpsc::WeakSender<PumpControl>,
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
        peers,
        snapshot_http,
        bootstrap_tx,
        ctrl_tx,
    } = deps;
    let pump_state = PumpState {
        tree: &tree,
        cursors: &cursors,
        tally: &tally,
        bootstrap: &bootstrap,
        bootstrap_tx: &bootstrap_tx,
        live_workers: &live_workers,
    };

    // Batches held back while their rank is `Pending`. Pump-local: the pump is
    // the only task that touches it, so no lock is needed.
    let mut held: HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>> = HashMap::new();
    // Ranks grafted from a snapshot whose continuity with the live stream is
    // not yet provable, mapped to the watermark and when the wait started.
    //
    // WHY deferred: a snapshot can be grafted before the rank's first live
    // batch has even arrived, so there is nothing to compare the watermark
    // against yet. The check runs on whichever batch turns up first — held or
    // live — and the entry is consumed by that one check, or by the sweep below
    // if no batch ever arrives.
    let mut awaiting_splice_proof: HashMap<KvWorkerId, PendingProof> = HashMap::new();
    let mut proof_sweep = tokio::time::interval(SPLICE_PROOF_SWEEP_INTERVAL);
    proof_sweep.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
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
                    Some(PumpControl::SpliceProbe {
                        rank,
                        epoch,
                        watermark,
                        verdict,
                    }) => {
                        // The rank may have proven itself, been forgotten,
                        // been re-registered, or been re-grafted by a gap retry
                        // while the probe was in flight. The epoch rules out
                        // another incarnation; the watermark rules out another
                        // graft of this one, which a retry makes under the same
                        // epoch.
                        let proof = match awaiting_splice_proof.get_mut(&rank) {
                            Some(proof)
                                if proof.watermark == watermark
                                    && bootstrap.epoch_of(&rank) == Some(epoch) =>
                            {
                                proof
                            }
                            _ => {
                                debug!(
                                    worker = ?rank,
                                    epoch,
                                    watermark,
                                    "kv-bootstrap: dropping a probe verdict that no longer \
                                     addresses live state",
                                );
                                continue;
                            }
                        };
                        match verdict {
                            SpliceVerdict::Advanced => {
                                awaiting_splice_proof.remove(&rank);
                                demote_unproven_rank(&pump_state, &mut held, &rank);
                            }
                            SpliceVerdict::NoAdvance => {
                                debug!(
                                    worker = ?rank,
                                    watermark,
                                    "kv-bootstrap: no peer is past the watermark; treating \
                                     the silent stream as continuous",
                                );
                                awaiting_splice_proof.remove(&rank);
                                bootstrap.record_rank_outcome(RankOutcome::Warm);
                            }
                            // No witness. Keep the rank warm and ask again — but
                            // not forever: a fleet that never answers (a
                            // single-replica deployment, say) would otherwise
                            // leave the verdict unresolved and the probe looping.
                            SpliceVerdict::Unknown => {
                                proof.unknown_probes += 1;
                                debug!(
                                    worker = ?rank,
                                    watermark = proof.watermark,
                                    probes = proof.unknown_probes,
                                    max_unknown_probes = MAX_UNKNOWN_PROBES,
                                    "kv-bootstrap: no witness answered this probe",
                                );
                                if proof.unknown_probes >= MAX_UNKNOWN_PROBES {
                                    info!(
                                        worker = ?rank,
                                        watermark = proof.watermark,
                                        probes = proof.unknown_probes,
                                        "kv-bootstrap: no peer could witness this rank's \
                                         progress; keeping the grafted state unproven",
                                    );
                                    awaiting_splice_proof.remove(&rank);
                                    bootstrap
                                        .record_rank_outcome(RankOutcome::WarmUnwitnessed);
                                }
                            }
                        }
                    }
                    None => {
                        debug!("kv-events pump: control channel closed");
                        ctrl_open = false;
                    }
                }
                continue;
            }
            // Ranks whose splice proof never arrived. Placed in the select rather
            // than keyed off event arrival BECAUSE the failure mode is the absence
            // of events: a rank that goes quiet right after a graft is exactly the
            // one that would otherwise never be checked.
            _ = proof_sweep.tick(), if !awaiting_splice_proof.is_empty() => {
                // Do NOT discard on silence alone. Reaching the deferred path
                // means nothing arrived between subscribing and grafting, and
                // the subscriber is live before the snapshot is fetched — so
                // silence is far more often "this rank published nothing" than
                // "we lost a delta". Discarding on a timer would throw away a
                // healthy warm tree on every quiet fleet, which is the exact
                // regression this feature exists to prevent. Ask the fleet
                // instead, and act only on positive evidence.
                let now = Instant::now();
                let mut targets = Vec::new();
                for (rank, proof) in awaiting_splice_proof
                    .iter_mut()
                    .filter(|(_, p)| p.due_for_probe(SPLICE_PROOF_TIMEOUT))
                {
                    // Re-armed before probing so retries are spaced by the
                    // timeout, never by the sweep tick.
                    proof.since = now;
                    // Forgotten: the `ForgetRanks` that follows drops the entry.
                    let Some(epoch) = bootstrap.epoch_of(rank) else {
                        continue;
                    };
                    targets.push(ProbeTarget {
                        rank: rank.clone(),
                        watermark: proof.watermark,
                        epoch,
                    });
                }
                // One pass for every due rank: a single cursor table per peer
                // answers all of them, so asking per rank would multiply the
                // fleet's work by the number of idle ranks.
                if !targets.is_empty() {
                    if let Some(ctrl_tx) = ctrl_tx.upgrade() {
                        spawn_splice_probe(
                            snapshot_http.clone(),
                            Arc::clone(&peers),
                            ctrl_tx,
                            targets,
                        );
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
                let proof = if awaiting_splice_proof.is_empty() {
                    None
                } else {
                    awaiting_splice_proof.remove(&worker)
                };
                if let Some(proof) = proof {
                    if leaves_gap(seq, proof.watermark) {
                        warn!(
                            worker = ?worker,
                            peer_cursor = proof.watermark,
                            first_live_seq = seq,
                            "kv-bootstrap: sequence gap between snapshot and live stream; \
                             discarding snapshot state for this rank to avoid stale cache entries",
                        );
                        // Held, not applied: `resolve_gap` either replays it after
                        // clearing, or keeps it for the retry's graft.
                        held.entry(worker.clone())
                            .or_default()
                            .push_back((seq, batch));
                        resolve_gap(&pump_state, &mut held, &worker);
                        continue;
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
    /// Obligation queue, so the graft path can hand a gapped rank back for
    /// another sweep. See [`resolve_gap`].
    bootstrap_tx: &'a mpsc::Sender<ObligationBatch>,
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

/// One rank a probe pass asks about, and the graft its verdict must still match
/// to be acted on (see [`PendingProof::watermark`]).
#[derive(Debug, Clone)]
struct ProbeTarget {
    rank: KvWorkerId,
    watermark: i64,
    epoch: u64,
}

/// Ask the fleet whether each target's publisher has moved past its watermark,
/// and post one verdict per target back to the pump.
///
/// Asks each peer for its cursor table alone, not a snapshot, and asks it ONCE
/// per pass for every target: the question is one integer per rank, and one
/// table answers it for all of them. The witness rules are on [`probe_fleet`].
fn spawn_splice_probe(
    http: reqwest::Client,
    peers: Arc<PeerRegistry>,
    ctrl_tx: mpsc::Sender<PumpControl>,
    targets: Vec<ProbeTarget>,
) {
    tokio::spawn(async move {
        let verdicts = probe_fleet(&http, &peers, &targets, SPLICE_PROBE_BUDGET).await;
        for (target, verdict) in targets.into_iter().zip(verdicts) {
            let msg = PumpControl::SpliceProbe {
                rank: target.rank,
                epoch: target.epoch,
                watermark: target.watermark,
                verdict,
            };
            if let Err(e) = ctrl_tx.send(msg).await {
                // Only the pump's shutdown closes the receiver, and a gone pump
                // has no proof state left to care about — but say so for
                // forensics.
                debug!(
                    error = %e,
                    "kv-bootstrap: control channel closed; dropping probe verdicts",
                );
                return;
            }
        }
    });
}

/// One pass over the fleet: fetch each peer's cursor table at most once, within
/// `budget`, and judge every target against every table that arrives.
///
/// Any peer's cursor is admissible evidence: sequence numbers are the
/// publisher's, so a peer reporting one above our watermark proves a batch we
/// never received was emitted. A peer too cold to bootstrap from is still a
/// valid witness, which is why this reads the wire cursor directly instead of
/// vetting.
///
/// A peer is a witness for a target only when its table names the rank AT OR
/// ABOVE the watermark. A decodable body alone is not one — a peer that never
/// saw the rank cannot say its publisher has not moved — and neither is a
/// cursor below the watermark: that peer has not observed the stream even up
/// to the point in question (a stalled subscription, or a reset we missed), so
/// it cannot testify about what came after. Either would otherwise manufacture
/// `NoAdvance` out of ignorance and tally the rank warm as if proven.
async fn probe_fleet(
    http: &reqwest::Client,
    peers: &PeerRegistry,
    targets: &[ProbeTarget],
    budget: Duration,
) -> Vec<SpliceVerdict> {
    let mut verdicts = vec![SpliceVerdict::Unknown; targets.len()];
    let deadline = Instant::now() + budget;
    for peer in peers.candidates() {
        if verdicts.iter().all(|v| *v == SpliceVerdict::Advanced) {
            break;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        let snap = match tokio::time::timeout(remaining, fetch_cursors(http, &peer)).await {
            Ok(Ok(Some(snap))) => snap,
            // Non-200 is already logged inside fetch_body; a read failure gets
            // one line here, so a peer serving corrupt bodies is not
            // indistinguishable from a peer that never saw the rank. Either way
            // this peer is not a witness for this pass.
            Ok(Ok(None)) => continue,
            Ok(Err(e)) => {
                debug!(
                    peer = %peer,
                    error = %format_args!("{e:#}"),
                    "kv-bootstrap: splice probe could not read this peer",
                );
                continue;
            }
            Err(_) => {
                debug!(
                    peer = %peer,
                    budget_ms = budget.as_millis(),
                    "kv-bootstrap: splice probe ran out of budget; peers not yet \
                     asked are not witnesses this pass",
                );
                break;
            }
        };
        // Same rule as vetting: a format this build does not recognise may give
        // the cursor table a meaning it does not know.
        if snap.format != SNAPSHOT_FORMAT {
            debug!(
                peer = %peer,
                got = snap.format,
                want = SNAPSHOT_FORMAT,
                "kv-bootstrap: splice probe ignoring a peer with an unknown snapshot format",
            );
            continue;
        }
        for (target, verdict) in targets.iter().zip(verdicts.iter_mut()) {
            if *verdict == SpliceVerdict::Advanced {
                continue;
            }
            match snap.wire_cursor_for(&target.rank.url, target.rank.dp_rank) {
                Some(seq) if seq > target.watermark => {
                    warn!(
                        worker = ?target.rank,
                        peer = %peer,
                        watermark = target.watermark,
                        peer_cursor = seq,
                        "kv-bootstrap: a peer is past our unproven watermark, so a \
                         batch we never received was published; discarding grafted \
                         state for this rank",
                    );
                    *verdict = SpliceVerdict::Advanced;
                }
                Some(seq) if seq == target.watermark => {
                    *verdict = SpliceVerdict::NoAdvance;
                }
                _ => {}
            }
        }
    }
    verdicts
}

/// Discard a graft whose live stream did not join its watermark, then hand the
/// rank back for one more sweep, or fail it cold when no retry is allowed.
///
/// A gap is the costliest failure: a snapshot was fetched, grafted, then thrown
/// away. A fresher snapshot usually splices, so the tracker allows one retry per
/// rank.
///
/// PRECONDITION: every live batch the rank has received is in `held` and none
/// has been applied. That is what lets a retry start from exactly the shape a
/// first attempt does — no tree state, no cursor, every delta held. Applying
/// the queue first would leave live deltas UNDER the retry's graft, whose seeded
/// cursor then filters their later removals as already reflected (a permanent
/// false hit on the success path), and any failure path's `clear_worker` would
/// wipe them.
///
/// The verdict is tallied only when final: a granted retry records nothing
/// here, and the retry's own resolution records the rank's one `RankOutcome`.
///
/// The retry is stamped now and marked [`LateJoin::Refused`], so only a sweep
/// that asks for an export newer than the gap can take it.
fn resolve_gap(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    rank: &KvWorkerId,
) {
    // `Failed` is the state a gap leaves, and the only one `retry_after_gap`
    // grants from. Set immediately before it, so `/readyz` has no room to read
    // a retried rank as terminal.
    st.bootstrap.set(rank, BootstrapState::Failed);
    if let Some(obligation) = st.bootstrap.retry_after_gap(rank) {
        let batch = ObligationBatch {
            obligations: vec![obligation],
            holding_since: Instant::now(),
            late_join: LateJoin::Refused,
        };
        match st.bootstrap_tx.try_send(batch) {
            Ok(()) => {
                // The discarded graft goes, untallied; `held` stays for the
                // retry's graft.
                st.tree.clear_worker(rank);
                st.cursors.lock().remove(rank);
                debug!(worker = ?rank, "kv-bootstrap: gapped rank re-queued for another sweep");
                return;
            }
            // A `Pending` rank nobody owns holds its batches until the
            // per-rank cap overflows, so fail it cold instead.
            Err(e) => {
                warn!("kv-bootstrap: could not re-queue gapped rank ({e}); leaving it cold");
            }
        }
    }
    discard_graft(st, rank, RankOutcome::Gap);
    for (seq, batch) in held.remove(rank).unwrap_or_default() {
        apply_batch(st.tree, st.cursors, st.tally, rank, seq, &batch);
    }
}

/// Resolve a gap a peer witnessed for a grafted rank that never proved its
/// splice, through the same [`resolve_gap`] the first-batch check uses.
///
/// Not [`fail_rank`]: that one only acts on a rank still
/// [`BootstrapState::Pending`], and this rank is `Recovered` — it was grafted
/// and is serving, so it holds no batches.
fn demote_unproven_rank(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    rank: &KvWorkerId,
) {
    // Only a live, still-`Recovered` rank is ours to demote.
    if st.bootstrap.state_of(rank) != Some(BootstrapState::Recovered)
        || !st.live_workers.lock().contains(rank)
    {
        return;
    }
    resolve_gap(st, held, rank);
}

/// Graft a vetted snapshot, seed cursors, then release held batches.
///
/// Runs on the pump so it is the sole tree writer for the duration.
fn apply_snapshot(
    st: &PumpState<'_>,
    held: &mut HashMap<KvWorkerId, VecDeque<(i64, KvEventBatch)>>,
    awaiting_splice_proof: &mut HashMap<KvWorkerId, PendingProof>,
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
                // `held` still holds the queue: `resolve_gap` replays it after
                // clearing, or keeps it for the retry's graft.
                resolve_gap(st, held, &rank);
                continue;
            }
            Some(_) => {}
            None => {
                awaiting_splice_proof
                    .insert(rank.clone(), PendingProof::new(peer_cursor, Instant::now()));
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
    use crate::state::kv_events::bootstrap::{CURSORS_ONLY_PARAM, SNAPSHOT_PATH};
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

    fn obligation(n: u32) -> ObligationBatch {
        discovery_batch(n, Instant::now())
    }

    fn discovery_batch(n: u32, holding_since: Instant) -> ObligationBatch {
        ObligationBatch {
            obligations: vec![(worker_id(&format!("http://w{n}:30000"), 0), n as u64)],
            holding_since,
            late_join: LateJoin::Permitted,
        }
    }

    /// A discovery burst is taken in one go, so one fleet-wide snapshot serves
    /// every rank instead of one fetch per worker.
    #[tokio::test]
    async fn take_pending_takes_the_whole_queued_burst() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        for n in 0..8 {
            tx.send(obligation(n)).await.unwrap();
        }
        let got = take_pending(&mut rx, &cancel).await.expect("obligations");
        assert_eq!(got.obligations.len(), 8, "all eight are taken together");
    }

    /// Absorbing a burst must adopt the STRICTEST freshness in it: the sweep
    /// fetches once for all of them, so a snapshot older than the last rank to
    /// start holding cannot splice for that rank.
    #[tokio::test]
    async fn take_pending_adopts_the_newest_holding_instant() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let base = Instant::now();
        let newest = base + Duration::from_millis(300);
        tx.send(discovery_batch(0, base)).await.unwrap();
        tx.send(discovery_batch(1, newest)).await.unwrap();
        tx.send(discovery_batch(2, base + Duration::from_millis(100)))
            .await
            .unwrap();

        let got = take_pending(&mut rx, &cancel).await.expect("obligations");
        assert_eq!(got.obligations.len(), 3);
        assert_eq!(
            got.freshness_floor, newest,
            "the floor must be the newest holding instant, not the first seen",
        );
    }

    /// It returns as soon as one obligation is queued, without debouncing.
    #[tokio::test]
    async fn take_pending_does_not_wait() {
        let (tx, mut rx) = mpsc::channel(8);
        let cancel = CancellationToken::new();
        tx.send(obligation(1)).await.unwrap();
        let started = Instant::now();
        let got = take_pending(&mut rx, &cancel).await.expect("obligations");
        let elapsed = started.elapsed();
        assert_eq!(got.obligations.len(), 1);
        assert!(
            elapsed < Duration::from_millis(50),
            "returned in {elapsed:?}; must not debounce",
        );
    }

    /// The load-bearing behaviour: ranks discovered WHILE a sweep is in flight
    /// are merged into that sweep's delivery, so they ride its snapshot rather
    /// than waiting for a second fetch.
    #[tokio::test]
    async fn drain_ready_merges_arrivals_from_during_the_sweep() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        tx.send(obligation(0)).await.unwrap();
        let mut pending = take_pending(&mut rx, &cancel).await.expect("obligations");
        assert_eq!(pending.obligations.len(), 1);

        // Stand in for the fetch: more workers show up while it is running.
        for n in 1..6 {
            tx.send(obligation(n)).await.unwrap();
        }
        let joined = drain_ready(&mut rx, &mut pending);
        assert_eq!(joined, 5, "late arrivals are reported");
        assert_eq!(
            pending.obligations.len(),
            6,
            "and merged into the same delivery",
        );
        assert!(
            pending.deferred.is_empty(),
            "discovery rides along rather than waiting for a sweep of its own",
        );
    }

    /// A gap retry landing mid-sweep must not be spent on that sweep's snapshot.
    #[tokio::test]
    async fn drain_ready_defers_a_gap_retry_the_sweep_cannot_speak_for() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let base = Instant::now();
        tx.send(discovery_batch(0, base)).await.unwrap();
        let mut pending = take_pending(&mut rx, &cancel).await.expect("obligations");

        let rank = worker_id("http://gapped:30000", 0);
        tx.send(ObligationBatch {
            obligations: vec![(rank.clone(), 7)],
            holding_since: base + Duration::from_millis(200),
            late_join: LateJoin::Refused,
        })
        .await
        .unwrap();

        let joined = drain_ready(&mut rx, &mut pending);
        assert_eq!(joined, 0, "a deferred batch is not counted as joined");
        assert_eq!(
            pending.obligations.len(),
            1,
            "and is not delivered into this sweep",
        );
        assert_eq!(pending.deferred.len(), 1, "it is handed to the next one");
        assert_eq!(pending.deferred[0].obligations[0].0, rank);
    }

    /// A rank merged in mid-sweep began holding after the frozen floor, so
    /// anything re-queued from this sweep must ask for at least its instant.
    /// The floor itself stays put: the request has already gone out under it.
    #[tokio::test]
    async fn late_admissions_raise_newest_holding_but_not_the_floor() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let base = Instant::now();
        tx.send(discovery_batch(0, base)).await.unwrap();
        let mut pending = take_pending(&mut rx, &cancel).await.expect("obligations");

        let late = base + Duration::from_secs(2);
        tx.send(discovery_batch(1, late)).await.unwrap();
        drain_ready(&mut rx, &mut pending);
        assert_eq!(
            pending.freshness_floor, base,
            "the floor is frozen once the sweep is in flight",
        );
        assert_eq!(pending.newest_holding, late);
    }

    /// Only obligations a graft could still discharge reach a sweep: a rank that
    /// already resolved, or whose incarnation is gone, would cost a fleet-wide
    /// fetch for nothing, and a duplicate would defeat its own coverage retry.
    #[test]
    fn retain_graftable_keeps_each_pending_incarnation_once() {
        let pending_rank = worker_id("http://pending:30000", 0);
        let resolved = worker_id("http://resolved:30000", 0);
        let reregistered = worker_id("http://reregistered:30000", 0);
        let ranks = [pending_rank.clone(), resolved.clone(), reregistered.clone()];
        let tracker = pending_tracker(&ranks);
        let obs = obligations(&tracker, &ranks);
        tracker.set(&resolved, BootstrapState::Recovered);
        let stale_epoch = tracker.epoch_of(&reregistered).unwrap();
        tracker.forget(std::slice::from_ref(&reregistered));
        tracker.register(std::slice::from_ref(&reregistered));

        let mut obligations = obs.clone();
        obligations.push(obs[0].clone());
        let mut sweep = PendingSweep::from(ObligationBatch {
            obligations,
            holding_since: Instant::now(),
            late_join: LateJoin::Permitted,
        });
        sweep.retain_graftable(&tracker);

        assert_eq!(
            sweep.obligations,
            vec![obs[0].clone()],
            "resolved, superseded (epoch {stale_epoch}) and duplicate obligations are dropped",
        );
    }

    /// The refusal is about freshness, not about being a retry: a retry the
    /// sweep's own floor already covers has nothing to gain from waiting.
    #[tokio::test]
    async fn drain_ready_admits_a_retry_the_floor_already_covers() {
        let (tx, mut rx) = mpsc::channel(64);
        let cancel = CancellationToken::new();
        let base = Instant::now();
        tx.send(discovery_batch(0, base)).await.unwrap();
        let mut pending = take_pending(&mut rx, &cancel).await.expect("obligations");

        tx.send(ObligationBatch {
            obligations: vec![(worker_id("http://early:30000", 0), 7)],
            holding_since: base - Duration::from_millis(200),
            late_join: LateJoin::Refused,
        })
        .await
        .unwrap();

        assert_eq!(drain_ready(&mut rx, &mut pending), 1);
        assert!(pending.deferred.is_empty());
    }

    /// One snapshot fetch never consumes the whole bootstrap deadline, so the
    /// sweep can reach another candidate. The deadlines span the configured
    /// default, where the half-deadline bound rather than the divisor binds.
    #[test]
    fn snapshot_fetch_timeout_is_a_strict_fraction_of_every_nonzero_deadline() {
        use crate::state::kv_events::bootstrap::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP as CAP;
        let default_secs = crate::config::DEFAULT_KV_BOOTSTRAP_TIMEOUT_MS / 1_000;
        assert_eq!(default_secs, 5, "test's premise: the default budget is 5s");

        for deadline_secs in [1, 2, default_secs, 6, 10, 20, 30, 60, 120, 600] {
            let deadline = Duration::from_secs(deadline_secs);
            let per_fetch = snapshot_fetch_timeout(deadline, CAP);
            assert!(
                per_fetch < deadline,
                "a single fetch may not consume the whole {deadline_secs}s deadline \
                 (got {per_fetch:?})",
            );
            assert!(
                deadline.as_secs_f64() / per_fetch.as_secs_f64() >= 2.0,
                "{deadline_secs}s must buy at least two attempts; one fetch got {per_fetch:?}",
            );
        }

        // Above the floor the divisor governs, so the budget buys the full
        // attempt count rather than merely two.
        assert_eq!(
            snapshot_fetch_timeout(Duration::from_secs(120), CAP)
                * SNAPSHOT_FETCH_ATTEMPTS_PER_DEADLINE,
            Duration::from_secs(120),
        );
        // The default budget derives a per-fetch bound below the floor, which
        // is the case the constructor warns about; pinned so a change to
        // either constant is deliberate.
        assert!(
            snapshot_fetch_timeout(Duration::from_secs(default_secs), CAP)
                < SNAPSHOT_FETCH_TIMEOUT_FLOOR,
            "the default budget cannot fund a snapshot-sized fetch; the \
             constructor's warning is what tells the operator so",
        );
        // The half-deadline bound binds here, so raising the cap changes nothing.
        assert_eq!(
            snapshot_fetch_timeout(Duration::from_secs(default_secs), CAP),
            snapshot_fetch_timeout(Duration::from_secs(default_secs), Duration::from_secs(600),),
        );
        // A zero deadline (bootstrap disabled) yields the floor, not a zero
        // timeout that would expire immediately.
        assert_eq!(
            snapshot_fetch_timeout(Duration::ZERO, CAP),
            SNAPSHOT_FETCH_TIMEOUT_FLOOR,
        );
    }

    /// The configurable cap binds a deadline that would otherwise derive
    /// above it, yields to storage-heavy fleets that need more per fetch,
    /// and never panics when misconfigured below the floor.
    #[test]
    fn snapshot_fetch_timeout_honours_the_configured_cap() {
        // A long deadline derives above the default cap, which holds it at
        // `CAP`; a configured cap binds at exactly its value.
        use crate::state::kv_events::bootstrap::DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP as CAP;
        assert_eq!(snapshot_fetch_timeout(Duration::from_secs(600), CAP), CAP);
        assert_eq!(
            snapshot_fetch_timeout(Duration::from_secs(600), Duration::from_secs(90)),
            Duration::from_secs(90),
            "the raised cap must not itself be exceeded",
        );
        assert_eq!(
            snapshot_fetch_timeout(Duration::from_secs(600), Duration::from_secs(400)),
            Duration::from_secs(150),
            "above the derivation the cap stops binding",
        );
        // A cap below the floor degrades to the cap itself rather than
        // panicking `clamp` — the CLI rejects this, but the math stays safe
        // for any path that skips it.
        assert_eq!(
            snapshot_fetch_timeout(Duration::from_secs(120), Duration::from_secs(2)),
            Duration::from_secs(2),
        );
        // And the default constant must not drift from the config default.
        assert_eq!(
            CAP,
            Duration::from_millis(crate::config::DEFAULT_KV_BOOTSTRAP_FETCH_TIMEOUT_CAP_MS),
        );
    }

    /// The CLI's minimum fetch cap is the per-fetch floor, so a validated cap
    /// never binds below it.
    #[test]
    fn snapshot_fetch_timeout_floor_matches_the_config_minimum_cap() {
        assert_eq!(
            SNAPSHOT_FETCH_TIMEOUT_FLOOR,
            Duration::from_millis(crate::config::MIN_KV_BOOTSTRAP_FETCH_TIMEOUT_CAP_MS),
        );
    }

    /// The producer reuses its cached export only for a caller whose freshness
    /// requirement it meets; an older export would predate the caller's
    /// subscription and gap on the watermark.
    #[tokio::test]
    async fn producer_reuses_its_export_only_when_it_meets_the_callers_max_age() {
        let index = KvEventIndex::new();
        let exported_at = || async {
            index
                .snapshot_cache
                .lock()
                .await
                .as_ref()
                .expect("an entry was cached")
                .exported_at
        };

        index.peer_snapshot_body(Duration::from_secs(60)).await;
        let first = exported_at().await;

        // A caller that can live with a minute-old tree gets the same one back.
        index.peer_snapshot_body(Duration::from_secs(60)).await;
        assert_eq!(first, exported_at().await, "a met requirement reuses");

        // A caller that began holding after that export cannot use it.
        tokio::time::sleep(Duration::from_millis(5)).await;
        index.peer_snapshot_body(Duration::ZERO).await;
        assert!(
            exported_at().await > first,
            "an unmet requirement must rebuild, not serve the stale export",
        );
    }

    /// The stamp has to be taken BEFORE the contents are sampled. Stamping
    /// completion would claim coverage the document does not have — exactly the
    /// hole the parameter closes — and the walk plus encode of a real tree is
    /// long enough for that to matter.
    #[tokio::test]
    async fn producer_stamps_the_export_before_it_samples() {
        let index = KvEventIndex::new();
        let before = Instant::now();
        index.peer_snapshot_body(Duration::ZERO).await;
        let after = Instant::now();
        let exported_at = index
            .snapshot_cache
            .lock()
            .await
            .as_ref()
            .expect("an entry was cached")
            .exported_at;
        assert!(
            exported_at >= before && exported_at <= after,
            "the stamp must sit inside the build, at its start",
        );
    }

    /// This body answers one integer per rank, so it must carry the cursors
    /// and NOT the tree. Asserting `nodes` is empty is also
    /// what keeps the body ungraftable — `from_wire` rejects an empty node list.
    #[tokio::test]
    async fn cursors_only_body_carries_cursors_and_no_nodes() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.set_bigram(false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));
        index.seed_stored_block_for_test(&worker_id("http://w1:30000", 0), 42, 111);

        let body = index.peer_cursors_body();
        let snap: PeerSnapshot = serde_json::from_slice(&body).expect("valid JSON");

        assert!(snap.nodes.is_empty(), "the tree must not be exported");
        assert_eq!(
            snap.wire_cursor_for("http://w1:30000", 0),
            Some(42),
            "the per-rank cursor must be answerable from this body",
        );
        assert_eq!(snap.block_size, 256);
        assert!(!snap.is_bigram);
        assert!(
            snap.producer_ready,
            "a settled replica holding nodes is a valid witness",
        );
    }

    /// A replica with an empty tree must not claim to be worth believing, for the
    /// same reason `snapshot_entry` checks it: a replica whose own bootstrap timed
    /// out is settled while holding nothing.
    #[tokio::test]
    async fn cursors_only_body_reports_not_ready_with_an_empty_tree() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.set_bigram(false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));

        let snap: PeerSnapshot =
            serde_json::from_slice(&index.peer_cursors_body()).expect("valid JSON");
        assert!(!snap.producer_ready, "an empty tree is not a source");
    }

    /// The cost claim, asserted structurally rather than by timing: serving cursors
    /// must leave the snapshot cache untouched, because populating it is the
    /// expensive walk this path exists to avoid.
    #[tokio::test]
    async fn cursors_only_body_never_populates_the_snapshot_cache() {
        let index = KvEventIndex::new();
        index.peer_cursors_body();
        assert!(
            index.snapshot_cache.lock().await.is_none(),
            "cursors-only must not walk or cache the tree",
        );
    }

    /// A peer that accepts and then never answers must not hold a pass past its
    /// budget: the pass ends with no witness rather than outliving the sweep.
    #[tokio::test]
    async fn probe_pass_gives_up_on_a_hung_peer_within_its_budget() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((sock, _)) = listener.accept().await {
                held.push(sock);
            }
        });
        let peers = PeerRegistry::new();
        peers.replace(vec![format!("http://{addr}")]);
        let targets = [ProbeTarget {
            rank: worker_id("http://w1:30000", 0),
            watermark: 10,
            epoch: 0,
        }];
        let verdicts = tokio::time::timeout(
            Duration::from_secs(5),
            probe_fleet(
                &reqwest::Client::new(),
                &peers,
                &targets,
                Duration::from_millis(200),
            ),
        )
        .await
        .expect("the budget, not the peer, must end the pass");
        assert_eq!(verdicts, vec![SpliceVerdict::Unknown]);
    }

    /// The two producer answers are intentionally NOT set-equal. A rank that
    /// observed a publisher and then lost the blocks (cleared here) keeps its
    /// cursor in the cursors-only body — that observation is still a valid
    /// witness — while the full export drops it, because its cursor table only
    /// covers ranks carrying tree nodes. The full side uses a SECOND rank that
    /// still carries a block, so the export is a real one (not the not-ready
    /// body an empty tree would fetch) and the assertion genuinely pins the
    /// carrier filtering. See [`KvEventIndex::peer_cursors_body`].
    #[tokio::test]
    async fn the_cursors_only_table_keeps_witnesses_the_full_export_loses() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.set_bigram(false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));
        let cleared = worker_id("http://w1:30000", 0);
        let holding = worker_id("http://w2:30000", 0);
        index.seed_stored_block_for_test(&holding, 7, 999);
        index.seed_stored_block_for_test(&cleared, 42, 111);
        // Observed-then-cleared: the cursor survives where the blocks do not.
        apply_batch(
            &index.tree,
            &index.cursors,
            &index.tally,
            &cleared,
            43,
            &batch(vec![KvCacheEvent::AllBlocksCleared]),
        );

        let thin: PeerSnapshot =
            serde_json::from_slice(&index.peer_cursors_body()).expect("valid JSON");
        assert_eq!(
            thin.wire_cursor_for("http://w1:30000", 0),
            Some(43),
            "a cleared rank is still a witness to its publisher's stream",
        );

        let full: PeerSnapshot =
            serde_json::from_slice(&index.peer_snapshot_body(Duration::ZERO).await.identity)
                .expect("valid JSON");
        assert!(
            !full.nodes.is_empty(),
            "the held block keeps this a real export, not the not-ready body",
        );
        assert_eq!(
            full.wire_cursor_for("http://w1:30000", 0),
            None,
            "carriers only: the cleared rank's cursor is dropped from the export",
        );
        assert_eq!(
            full.wire_cursor_for("http://w2:30000", 0),
            Some(7),
            "the rank still carrying blocks keeps its cursor in the export",
        );
    }

    /// A witness body: no nodes, just a cursor table, as the cursors-only
    /// producer serves it. With no entries it is a cold peer's answer.
    fn witness_snapshot(entries: &[(&str, u32, i64)]) -> PeerSnapshot {
        PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size: 64,
            is_bigram: false,
            producer_ready: false,
            workers: entries
                .iter()
                .map(|(url, dp_rank, _)| WireWorker {
                    url: (*url).to_string(),
                    dp_rank: *dp_rank,
                })
                .collect(),
            cursors: entries
                .iter()
                .enumerate()
                .map(|(i, (_, _, seq))| (i as u32, *seq))
                .collect(),
            nodes: Vec::new(),
        }
    }

    /// Run one probe pass against a one-peer fleet serving `snap`, returning
    /// the verdicts the probe posts to the pump-control channel — the same path
    /// a verdict takes in production — in target order. Asserts along the way
    /// that the peer was asked exactly once, for the cursor table alone.
    async fn probe_pass(snap: PeerSnapshot, probed: &[(KvWorkerId, i64)]) -> Vec<SpliceVerdict> {
        let (base, queries) = serve_snapshot_sequence(vec![snap]).await;
        let peers = Arc::new(PeerRegistry::new());
        peers.replace(vec![base]);
        let (tx, mut rx) = mpsc::channel(probed.len());
        let targets = probed
            .iter()
            .map(|(rank, watermark)| ProbeTarget {
                rank: rank.clone(),
                watermark: *watermark,
                epoch: 0,
            })
            .collect();
        spawn_splice_probe(reqwest::Client::new(), peers, tx, targets);
        let mut verdicts = Vec::new();
        for (rank, watermark) in probed {
            let outcome = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("a live one-peer fleet always gets an answer")
                .expect("the control channel outlives the probe");
            match outcome {
                PumpControl::SpliceProbe {
                    rank: got,
                    watermark: got_watermark,
                    verdict,
                    ..
                } => {
                    assert_eq!((&got, got_watermark), (rank, *watermark));
                    verdicts.push(verdict);
                }
                other => panic!("expected a splice-probe verdict, got {other:?}"),
            }
        }
        let queries = queries.lock().expect("queries lock");
        assert_eq!(queries.len(), 1, "one fetch per peer answers every target");
        assert!(
            queries.iter().all(|q| q
                .as_deref()
                .unwrap_or_default()
                .contains(CURSORS_ONLY_PARAM)),
            "every probe request must ask for the cursor table alone",
        );
        verdicts
    }

    async fn probe_once(snap: PeerSnapshot, probed: &KvWorkerId, watermark: i64) -> SpliceVerdict {
        probe_pass(snap, &[(probed.clone(), watermark)]).await[0]
    }

    /// A peer whose cursor for the rank is BELOW the watermark has not seen the
    /// stream even that far, so it cannot say the publisher stopped there.
    #[tokio::test]
    async fn probe_ignores_a_witness_below_the_watermark() {
        let snap = witness_snapshot(&[("http://w1:30000", 0, 4)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::Unknown,
        );
    }

    /// Every due rank is judged from the same table: one request, one verdict
    /// per rank.
    #[tokio::test]
    async fn probe_pass_judges_every_target_from_one_fetch() {
        let snap = witness_snapshot(&[("http://w1:30000", 0, 11), ("http://w2:30000", 0, 7)]);
        assert_eq!(
            probe_pass(
                snap,
                &[
                    (worker_id("http://w1:30000", 0), 10),
                    (worker_id("http://w2:30000", 0), 7),
                ],
            )
            .await,
            vec![SpliceVerdict::Advanced, SpliceVerdict::NoAdvance],
        );
    }

    /// A peer whose cursor table does not NAME the probed rank has never
    /// observed its publisher, so its body alone must not resolve the verdict
    /// to NoAdvance.
    #[tokio::test]
    async fn probe_ignores_a_body_that_does_not_name_the_rank() {
        let snap = witness_snapshot(&[("http://other:30000", 0, 999)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::Unknown,
        );
    }

    #[tokio::test]
    async fn probe_reports_no_advance_when_the_best_witness_is_at_the_watermark() {
        let snap = witness_snapshot(&[("http://w1:30000", 0, 10)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::NoAdvance,
        );
    }

    #[tokio::test]
    async fn probe_reports_advance_when_a_witness_is_past_the_watermark() {
        let snap = witness_snapshot(&[("http://w1:30000", 0, 11)]);
        assert_eq!(
            probe_once(snap, &worker_id("http://w1:30000", 0), 10).await,
            SpliceVerdict::Advanced,
        );
    }

    // ---- sweep early exit (SweepResult::FleetCold) ----

    /// A replica that has published a block size but not yet its hashing mode
    /// must not vet anything. That window is real — `add_worker` writes the two
    /// in sequence — and a peer's tree vetted inside it is compared against an
    /// identity this replica has not established, so every grafted block is one
    /// it can never match.
    #[tokio::test]
    async fn the_sweep_refuses_to_vet_on_a_half_published_hash_config() {
        let deps = sweep_deps(vec!["http://peer:30000".into()], 64, &["http://w1:30000"]);
        // Undo the mode half that `sweep_deps` publishes, keeping the size.
        let half = BlockSizeOracle::new();
        half.try_set(64).unwrap();
        assert_eq!(half.get(), Some(64));
        assert_eq!(half.hash_config(), None);
        let deps = BootstrapDeps {
            oracle: half,
            ..deps
        };

        let ctx = SweepCtx {
            http: &deps.http,
            peers: &deps.peers,
            bootstrap: &deps.bootstrap,
            live_workers: &deps.live_workers,
            oracle: &deps.oracle,
            freshness_floor: Instant::now(),
        };
        let mut state = SweepState::new();
        let reason = Mutex::new(None);
        let pass = sweep_peers(
            &ctx,
            &[worker_id("http://w1:30000", 0)],
            &mut state,
            &reason,
        )
        .await;
        assert!(
            matches!(pass, SweepPass::KeepLooking),
            "no peer may be consulted before the hashing identity is established",
        );
        assert_eq!(
            deps.bootstrap.peer_outcome_counts(),
            vec![],
            "and no fetch was attempted, so no peer outcome is tallied",
        );
    }

    fn sweep_deps(peers: Vec<String>, block_size: u32, live: &[&str]) -> BootstrapDeps {
        sweep_deps_with_ctrl(peers, block_size, live).0
    }

    /// [`sweep_deps`], keeping the pump end of the control channel.
    fn sweep_deps_with_ctrl(
        peers: Vec<String>,
        block_size: u32,
        live: &[&str],
    ) -> (BootstrapDeps, mpsc::Receiver<PumpControl>) {
        let registry = Arc::new(PeerRegistry::new());
        registry.replace(peers);
        let oracle = BlockSizeOracle::new();
        oracle.try_set(block_size).expect("first set establishes");
        // `hash_config` is both-or-neither, and the sweep refuses to vet
        // without it.
        oracle.set_bigram(false);
        let (ctrl_tx, ctrl_rx) = mpsc::channel(8);
        let deps = BootstrapDeps {
            http: reqwest::Client::new(),
            peers: registry,
            bootstrap: Arc::new(BootstrapTracker::new(Duration::from_secs(3600))),
            live_workers: Arc::new(Mutex::new(live.iter().map(|u| worker_id(u, 0)).collect())),
            oracle,
            ctrl_tx,
        };
        (deps, ctrl_rx)
    }

    /// A snapshot with real content, carried by `carrier` only.
    fn warm_snapshot(carrier: &str, block_size: u32) -> PeerSnapshot {
        PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size,
            is_bigram: false,
            producer_ready: true,
            workers: vec![WireWorker {
                url: carrier.into(),
                dp_rank: 0,
            }],
            cursors: vec![(0, 5)],
            nodes: vec![SnapshotNode {
                parent: None,
                block_hash: 111,
                workers: vec![0],
                tiers: vec![],
            }],
        }
    }

    /// A fleet whose siblings all hold empty trees settles cold at the end of
    /// the first pass instead of holding readiness until the deadline.
    #[tokio::test]
    async fn sweep_settles_early_when_every_peer_is_cold() {
        let (p1, q1) = serve_snapshot_sequence(vec![witness_snapshot(&[])]).await;
        let (p2, q2) = serve_snapshot_sequence(vec![witness_snapshot(&[])]).await;
        let deps = sweep_deps(vec![p1, p2], 64, &["http://w1:30000"]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        // A 3600s deadline would hang a broken test; the outer timeout fails
        // fast instead, and reaching it IS the regression.
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(&deps, &ranks, Duration::from_secs(3600), Instant::now()),
        )
        .await
        .expect("a cold fleet settles immediately, not at the deadline");
        assert!(
            matches!(result, SweepResult::FleetCold { peers_tried: 2 }),
            "every peer proved empty, so the sweep must not wait out the deadline",
        );
        // One fetch per peer: the verdict lands at the end of the FIRST pass,
        // not after a cooldown round of re-fetches.
        assert_eq!(q1.lock().expect("queries lock").len(), 1);
        assert_eq!(q2.lock().expect("queries lock").len(), 1);
    }

    /// An unreachable peer says nothing about its warmth — it may be a warm
    /// sibling mid-restart — so its presence must veto the early cold exit.
    #[tokio::test]
    async fn sweep_waits_out_the_deadline_when_a_peer_is_unreachable() {
        let (cold, _q) = serve_snapshot_sequence(vec![witness_snapshot(&[])]).await;
        // Nothing listens on port 1: a fast connection-refused, not a hang.
        let deps = sweep_deps(
            vec![cold, "http://127.0.0.1:1".into()],
            64,
            &["http://w1:30000"],
        );
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result =
            sweep_until_deadline(&deps, &ranks, Duration::from_millis(500), Instant::now()).await;
        assert!(
            matches!(result, SweepResult::TimedOut { peers_tried: 2, .. }),
            "an unanswered peer keeps the sweep waiting until the deadline",
        );
    }

    /// A warm peer that holds no blocks for the bootstrapping ranks is still a
    /// warm peer: the fleet HAS state, so the sweep keeps looking rather than
    /// declaring the fleet cold. And the answer is still an attempt: tallied,
    /// and named as the reason the deadline ran out.
    #[tokio::test]
    async fn sweep_waits_out_the_deadline_when_a_peer_is_warm_but_uncovering() {
        let (warm, q) = serve_snapshot_sequence(vec![warm_snapshot("http://w2:30000", 64)]).await;
        let deps = sweep_deps(vec![warm], 64, &["http://w1:30000", "http://w2:30000"]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result =
            sweep_until_deadline(&deps, &ranks, Duration::from_millis(500), Instant::now()).await;
        let SweepResult::TimedOut {
            peers_tried: 1,
            last_reason,
        } = result
        else {
            panic!("a warm peer keeps the sweep waiting even when it covers nothing we need");
        };
        assert!(
            last_reason
                .as_deref()
                .is_some_and(|r| r.contains("tracks none of the ranks")),
            "the timeout must name why the answering peer was not used; got {last_reason:?}",
        );
        assert!(
            deps.bootstrap
                .peer_outcome_counts()
                .iter()
                .any(|(label, n)| *label == "cold_peer" && *n >= 1),
            "a non-covering answer is a fetch attempt and must be tallied; got {:?}",
            deps.bootstrap.peer_outcome_counts(),
        );
        assert!(
            q.lock().expect("queries lock").len() <= 2,
            "a non-covering answer cools the peer down; it is not re-fetched every pass",
        );
    }

    /// The manager re-runs `add_worker` for a worker it already knows (its
    /// reconcile loop does so for a worker that advertises no model name), so a
    /// rank the tracker already holds must not buy another sweep: a `Pending`
    /// rank has one in flight, and a resolved one has nothing left to fetch.
    /// Only a rank `remove_worker` forgot registers afresh.
    #[tokio::test]
    async fn register_for_bootstrap_skips_ranks_the_tracker_already_holds() {
        let index = KvEventIndex::new_with_bootstrap(
            reqwest::Client::new(),
            BlockSizeOracle::new(),
            Arc::new(BootstrapTracker::new(Duration::from_secs(3600))),
        );
        let id = worker_id("http://w1:30000", 0);
        let ranks = std::slice::from_ref(&id);

        let first = index.register_for_bootstrap(ranks);
        assert_eq!(first.len(), 1, "an untracked rank is registered");
        assert!(
            index.register_for_bootstrap(ranks).is_empty(),
            "a Pending rank already has a sweep",
        );
        index.bootstrap.set(&id, BootstrapState::Failed);
        assert!(
            index.register_for_bootstrap(ranks).is_empty(),
            "a resolved rank has nothing left to fetch",
        );
        assert_eq!(index.bootstrap.state_of(&id), Some(BootstrapState::Failed));

        index.bootstrap.forget(ranks);
        let again = index.register_for_bootstrap(ranks);
        assert_eq!(again.len(), 1, "a forgotten rank registers afresh");
        assert_ne!(again[0].1, first[0].1, "as a new incarnation");

        assert!(
            KvEventIndex::new().register_for_bootstrap(ranks).is_empty(),
            "a disabled tracker registers nothing",
        );
    }

    /// Permanent rejection (here a block-size mismatch) is as terminal as an
    /// empty tree: retrying cannot change the answer, so a fleet of nothing
    /// but incompatible peers also settles early.
    #[tokio::test]
    async fn sweep_settles_early_when_every_peer_is_permanently_rejected() {
        let (p, q) = serve_snapshot_sequence(vec![warm_snapshot("http://w1:30000", 999)]).await;
        let deps = sweep_deps(vec![p], 64, &["http://w1:30000"]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(&deps, &ranks, Duration::from_secs(3600), Instant::now()),
        )
        .await
        .expect("an incompatible fleet settles immediately, not at the deadline");
        assert!(
            matches!(result, SweepResult::FleetCold { peers_tried: 1 }),
            "permanent rejection must count toward the all-cold verdict",
        );
        assert_eq!(
            q.lock().expect("queries lock").len(),
            1,
            "a permanently rejected peer is never re-fetched",
        );
    }

    /// An empty candidate set is not a cold fleet: `all()` on an empty
    /// iterator is vacuously true, and the peer set legitimately dips to
    /// empty during EndpointSlice repacks — exactly the race the retry loop
    /// exists for (worker discovery wins against the peer watch). This
    /// registry HAS seen peers, so `known_to_have_no_peers` stays false and
    /// only the verdict's own `is_empty` guard stands between this and a
    /// wrong `FleetCold`.
    #[tokio::test]
    async fn sweep_does_not_settle_cold_on_a_transiently_empty_peer_set() {
        let deps = sweep_deps(vec!["http://127.0.0.1:1".into()], 64, &["http://w1:30000"]);
        deps.peers.replace(vec![]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result =
            sweep_until_deadline(&deps, &ranks, Duration::from_millis(500), Instant::now()).await;
        assert!(
            matches!(result, SweepResult::TimedOut { .. }),
            "an empty candidate set is no information, not a cold fleet",
        );
    }

    /// A registry that never had peers maps to `NoPeers`, not `FleetCold`:
    /// the two deliver identically but log differently, and "no siblings
    /// exist" is a different operational fact from "siblings proved empty".
    #[tokio::test]
    async fn sweep_reports_no_peers_when_discovery_confirms_none() {
        let deps = sweep_deps(vec![], 64, &["http://w1:30000"]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(&deps, &ranks, Duration::from_secs(3600), Instant::now()),
        )
        .await
        .expect("a confirmed-empty discovery settles immediately");
        assert!(matches!(result, SweepResult::NoPeers));
    }

    /// A peer that warms up mid-sweep must be FOUND, not FleetColded on a
    /// stale cold classification. The unreachable second peer is what keeps
    /// the sweep alive past the cold peer's cooldown: without an unclassified
    /// candidate in the mix, the fleet settles cold at the end of the pass
    /// that classifies it — before its cooldown ever lets a warmer answer
    /// through — by design.
    #[tokio::test]
    async fn sweep_finds_a_peer_that_warms_during_the_sweep() {
        let warm = warm_snapshot("http://w1:30000", 64);
        let (flipper, _q) = serve_snapshot_sequence(vec![witness_snapshot(&[]), warm]).await;
        let deps = sweep_deps(
            vec![flipper, "http://127.0.0.1:1".into()],
            64,
            &["http://w1:30000"],
        );
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(&deps, &ranks, Duration::from_secs(3600), Instant::now()),
        )
        .await
        .expect("the warm answer ends the sweep on its refetch pass");
        assert!(
            matches!(result, SweepResult::Found(_)),
            "a peer that warms mid-sweep must be found, not settled cold",
        );
    }

    /// The cold→warm remove itself: a peer re-fetched as warm-but-uncovering
    /// must drop its cold witness, or its stale classification would let the
    /// fleet settle cold while a warm peer exists. Drives `sweep_peers`
    /// directly so the refetch is not hostage to pass timing.
    #[tokio::test]
    async fn a_warm_answer_erases_a_peers_cold_witness() {
        let uncovering = warm_snapshot("http://w2:30000", 64);
        let (flipper, _q) = serve_snapshot_sequence(vec![witness_snapshot(&[]), uncovering]).await;
        // The unreachable second candidate keeps each pass from settling
        // cold on the flipper's classification alone.
        let deps = sweep_deps(
            vec![flipper.clone(), "http://127.0.0.1:1".into()],
            64,
            &["http://w1:30000", "http://w2:30000"],
        );
        let ctx = SweepCtx {
            http: &deps.http,
            peers: &deps.peers,
            bootstrap: &deps.bootstrap,
            live_workers: &deps.live_workers,
            oracle: &deps.oracle,
            freshness_floor: Instant::now(),
        };
        let last_reason = Mutex::new(None);
        let mut state = SweepState::new();
        let ranks = vec![worker_id("http://w1:30000", 0)];

        let pass1 = sweep_peers(&ctx, &ranks, &mut state, &last_reason).await;
        assert!(matches!(pass1, SweepPass::KeepLooking));
        assert!(state.cold_witnessed.contains(&flipper));

        // Skip the cooldown so pass 2 re-fetches immediately.
        state.cooldown.clear();
        let pass2 = sweep_peers(&ctx, &ranks, &mut state, &last_reason).await;
        assert!(matches!(pass2, SweepPass::KeepLooking));
        assert!(
            !state.cold_witnessed.contains(&flipper),
            "the warm answer must erase the cold witness",
        );
        // And with the witness gone, a fleet of {warm-uncovering, rejected}
        // is NOT cold — the warm peer might cover the ranks later.
        state.note_permanent_reject("http://c:30000");
        assert!(!state.fleet_is_cold(&[flipper.clone(), "http://c:30000".into()]));
    }

    /// The verdict primitives, pinned directly: cold counts, unknown vetoes,
    /// rejection counts, empty is not cold.
    #[test]
    fn fleet_is_cold_only_when_every_candidate_proved_hopeless() {
        let mut state = SweepState::new();
        let a = "http://a:30000".to_string();
        let cold = witness_snapshot(&[]);
        let warm = warm_snapshot("http://w1:30000", 64);
        let candidates = std::slice::from_ref(&a);

        // No information at all is not cold.
        assert!(!state.fleet_is_cold(&[]));
        assert!(!state.fleet_is_cold(candidates));

        state.note_answer(&a, &cold);
        assert!(state.fleet_is_cold(candidates));

        // An unreachable spell erases the witness: unknown warmth vetoes.
        state.note_unreachable(&a);
        assert!(!state.fleet_is_cold(candidates));

        // Re-cold, then a warm answer erases it again.
        state.note_answer(&a, &cold);
        state.note_answer(&a, &warm);
        assert!(!state.fleet_is_cold(candidates));

        // Settled-empty (ready flag set, zero nodes) is a cold witness too:
        // `holds_no_state` looks only at nodes.
        let settled_empty = PeerSnapshot {
            producer_ready: true,
            nodes: vec![],
            ..warm_snapshot("http://a:30000", 64)
        };
        state.note_answer(&a, &settled_empty);
        assert!(state.fleet_is_cold(candidates));

        // A peer still bootstrapping reports `producer_ready: false` while
        // holding nodes: vetting refuses it, but it is not a cold witness.
        let mid_bootstrap = PeerSnapshot {
            producer_ready: false,
            ..warm_snapshot("http://w1:30000", 64)
        };
        assert!(mid_bootstrap.is_cold(), "vetting still refuses to graft it");
        state.note_answer(&a, &mid_bootstrap);
        assert!(
            !state.fleet_is_cold(candidates),
            "a peer still bootstrapping with a non-empty tree is not a cold witness",
        );

        // Permanent rejection is terminal on its own.
        let mut state = SweepState::new();
        state.note_permanent_reject(&a);
        assert!(state.fleet_is_cold(candidates));
    }

    /// End to end for the same hazard: the only candidate is a sibling
    /// mid-bootstrap that already holds blocks. The sweep must keep looking
    /// (and find it once it settles), not settle cold on the first pass.
    #[tokio::test]
    async fn sweep_waits_for_a_sibling_that_is_still_bootstrapping() {
        let mid_bootstrap = PeerSnapshot {
            producer_ready: false,
            ..warm_snapshot("http://w1:30000", 64)
        };
        let (peer, _q) =
            serve_snapshot_sequence(vec![mid_bootstrap, warm_snapshot("http://w1:30000", 64)])
                .await;
        let deps = sweep_deps(vec![peer], 64, &["http://w1:30000"]);
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            sweep_until_deadline(&deps, &ranks, Duration::from_secs(3600), Instant::now()),
        )
        .await
        .expect("the peer settles on its second answer, well inside the deadline");
        assert!(
            matches!(result, SweepResult::Found(_)),
            "a peer that holds a tree but has not settled must keep the sweep alive",
        );
    }

    /// Cooldown sits a peer out for exactly the configured passes, then lets
    /// it be refetched — a peer whose entry reaches zero is never stuck.
    #[test]
    fn cooling_sits_out_exactly_the_configured_passes() {
        let mut state = SweepState::new();
        state.note_sit_out("http://a:30000");
        for _ in 0..PEER_COOLDOWN_PASSES {
            assert!(state.cooling("http://a:30000"));
        }
        assert!(
            !state.cooling("http://a:30000"),
            "a peer at zero is refetched, never stuck",
        );
        assert!(!state.cooling("http://never-seen:30000"));
    }

    /// A same-length membership swap mid-pass — one pod replaced at constant
    /// replica count, the rolling-update norm — must veto the cold-fleet
    /// verdict even though the registry LENGTH is unchanged: the stale
    /// candidate list proved cold, but the live newcomer was never consulted.
    /// The swapping server stands in for the EndpointSlice informer landing a
    /// `replace` while the pass was in flight.
    #[tokio::test]
    async fn sweep_survives_a_same_length_membership_swap_mid_pass() {
        let cold = witness_snapshot(&[]);
        let registry = Arc::new(PeerRegistry::new());
        let reg_in_handler = Arc::clone(&registry);
        let app = axum::Router::new().route(
            SNAPSHOT_PATH,
            axum::routing::get(move || {
                let registry = Arc::clone(&reg_in_handler);
                let snap = cold.clone();
                async move {
                    // First fetch: the informer swaps this peer for one that
                    // is down — same length, different membership.
                    registry.replace(vec!["http://127.0.0.1:1".to_string()]);
                    axum::Json(snap)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        registry.replace(vec![format!("http://{addr}")]);

        let deps = BootstrapDeps {
            peers: registry,
            ..sweep_deps(vec![], 64, &["http://w1:30000"])
        };
        let ranks = vec![worker_id("http://w1:30000", 0)];
        let result =
            sweep_until_deadline(&deps, &ranks, Duration::from_millis(500), Instant::now()).await;
        assert!(
            matches!(result, SweepResult::TimedOut { .. }),
            "a mid-pass membership swap must veto the verdict, length unchanged or not",
        );
    }

    /// `deliver_bootstrap` is the one recording point for the sweep metric:
    /// every terminal verdict tallies exactly once, so `fleet_cold` stays
    /// distinguishable from `timed_out` on dashboards.
    #[tokio::test]
    async fn deliver_bootstrap_records_the_sweep_verdict_once() {
        let (deps, mut ctrl_rx) = sweep_deps_with_ctrl(vec![], 64, &[]);
        let bootstrap = Arc::clone(&deps.bootstrap);
        deliver_bootstrap(
            &deps,
            vec![],
            SweepResult::FleetCold { peers_tried: 3 },
            Duration::from_secs(5),
        )
        .await;
        assert!(
            matches!(
                ctrl_rx.recv().await,
                Some(PumpControl::AbandonBootstrap { .. })
            ),
            "a cold fleet releases its ranks through the same abandon path",
        );
        assert!(
            bootstrap.sweep_result_counts().contains(&("fleet_cold", 1)),
            "the verdict must be tallied exactly once; got {:?}",
            bootstrap.sweep_result_counts(),
        );
    }

    /// Serve a sequence of canned bodies on the real snapshot path: hit N
    /// gets `snaps[min(N, len-1)]`, so a test can flip a peer's temperature
    /// mid-sweep.
    async fn serve_snapshot_sequence(
        snaps: Vec<PeerSnapshot>,
    ) -> (String, Arc<std::sync::Mutex<Vec<Option<String>>>>) {
        let queries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = Arc::clone(&queries);
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let app = axum::Router::new().route(
            SNAPSHOT_PATH,
            axum::routing::get(move |uri: axum::http::Uri| {
                let seen = Arc::clone(&seen);
                let hits = Arc::clone(&hits);
                let snaps = snaps.clone();
                async move {
                    seen.lock()
                        .expect("queries lock")
                        .push(uri.query().map(str::to_string));
                    let n = hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    axum::Json(snaps[n.min(snaps.len() - 1)].clone())
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), queries)
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

    /// The premise behind `BootstrapDeps::deadline`'s settled branch: the
    /// remaining window saturates at zero, so a late rank needs the configured
    /// timeout instead.
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

    /// Nothing queued means nothing added, and no spinning.
    #[tokio::test]
    async fn drain_ready_on_empty_queue_adds_nothing() {
        let (_tx, mut rx) = mpsc::channel::<ObligationBatch>(4);
        let mut pending = PendingSweep::from(obligation(0));
        assert_eq!(drain_ready(&mut rx, &mut pending), 0);
        assert_eq!(pending.obligations.len(), 1);
    }

    /// Cancellation while idle ends the coordinator rather than parking forever.
    #[tokio::test]
    async fn take_pending_stops_on_cancel() {
        let (_tx, mut rx) = mpsc::channel::<ObligationBatch>(4);
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert!(
            take_pending(&mut rx, &cancel).await.is_none(),
            "cancelled coordinator stops",
        );
    }

    /// All senders dropped with nothing pending ends the coordinator too.
    #[tokio::test]
    async fn take_pending_stops_when_senders_drop() {
        let (tx, mut rx) = mpsc::channel::<ObligationBatch>(4);
        let cancel = CancellationToken::new();
        drop(tx);
        assert!(
            take_pending(&mut rx, &cancel).await.is_none(),
            "closed channel stops the coordinator",
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
        /// Obligations the pump handed back, e.g. a gap-driven retry.
        bootstrap_rx: mpsc::Receiver<ObligationBatch>,
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
        // Real queue so a gap-driven re-queue is observable rather than dropped.
        let (bootstrap_tx, bootstrap_rx) = mpsc::channel(16);
        let tally = Arc::new(EventTally::new());
        let pump = tokio::spawn(pump_loop(
            PumpDeps {
                tally: Arc::clone(&tally),
                tree: tree.clone(),
                engine_reported_load: engine_reported_load.clone(),
                cursors: cursors.clone(),
                live_workers: live_set.clone(),
                bootstrap: bootstrap.clone(),
                // Empty peer set: a splice probe finds no witness and returns
                // `Unknown`, so these tests exercise the pump's own gates without
                // any network. Probe verdicts are driven directly instead.
                peers: Arc::new(PeerRegistry::new()),
                snapshot_http: reqwest::Client::new(),
                bootstrap_tx: bootstrap_tx.clone(),
                ctrl_tx: ctrl_tx.downgrade(),
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
            bootstrap_rx,
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

        // A gapped rank is handed back for its one retry, so it is Pending.
        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "snapshot state must be discarded for a gapped rank",
        );
        // The live batch is held for the retry's graft, like any Pending rank's;
        // `abandoned_gap_retry_replays_every_held_batch` covers its release.
        assert_eq!(h.tree.match_prefix(None, &[42]).matched_blocks, 0);
        assert!(h.cursors.lock().get(&id).is_none());
        // Readiness waits: the gapped rank is Pending for its retry, so the
        // tracker is unsettled until that resolves or the deadline expires.
        // Warming the tree is preferred over opening `/readyz` on a cold rank.
        assert!(!tracker.settled());
    }

    /// A gap is the costliest failure — a snapshot was fetched, grafted, then
    /// thrown away. So the rank is handed back for one more sweep instead of
    /// staying cold with budget unspent. The one-retry cap is covered by
    /// `a_second_gap_fails_the_rank_cold_and_replays_its_batches`.
    #[tokio::test]
    async fn pump_requeues_a_gapped_rank_once() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let mut h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // Watermark 5, first live batch seq 9 — 6..8 lost, so this gaps.
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

        let requeued = h.bootstrap_rx.try_recv().expect("gapped rank re-queued");
        assert_eq!(requeued.obligations.len(), 1);
        assert_eq!(
            requeued.obligations[0].0, id,
            "the gapped rank itself is handed back",
        );
        assert_eq!(
            requeued.late_join,
            LateJoin::Refused,
            "a retry must not be spent on a snapshot fetched before the gap",
        );
        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "back to Pending so the next sweep may graft onto it",
        );
    }

    /// A retried rank must reach its second graft in the same shape as its
    /// first: nothing applied, every delta held. If the first attempt's held
    /// batches were applied before the retry, the retry's graft lands on top of
    /// them and its seeded cursor then filters their later removals as already
    /// reflected — so a block the engine has evicted stays attributed to the
    /// rank for good.
    #[tokio::test]
    async fn gap_retry_graft_does_not_resurrect_a_block_the_stream_removed() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let mut h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        // seq 7 stores X. Held, because the rank is Pending.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 7,
            batch: batch(vec![stored(None, vec![500])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        // First graft watermarked 5: seq 6 is missing, so this gaps and retries.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        let retry = h.bootstrap_rx.recv().await.expect("gapped rank re-queued");

        // seq 8 evicts X while the retry is in flight.
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 8,
            batch: batch(vec![KvCacheEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![500],
                medium: None,
            })]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        // The retry's peer had applied both, so its export no longer holds X.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: retry.obligations,
                vetted: Box::new(vetted_for(&id, 8)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert!(
            !h.tree.match_prefix(None, &[500]).workers().contains(&id),
            "the engine evicted X at seq 8; the rank must not still own it",
        );
        assert_eq!(rank_count(&tracker, "warm"), 1);
        assert_eq!(
            rank_count(&tracker, "gap"),
            0,
            "a retried gap is not a verdict; the retry's outcome is the rank's one count",
        );
    }

    /// A retry that fails must not cost the rank the deltas it held before it:
    /// abandoning replays every batch the rank has received, both attempts'.
    #[tokio::test]
    async fn abandoned_gap_retry_replays_every_held_batch() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        let mut h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 7,
            batch: batch(vec![stored(None, vec![500])]),
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
        let retry = h.bootstrap_rx.recv().await.expect("gapped rank re-queued");
        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 8,
            batch: batch(vec![stored(None, vec![600])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::AbandonBootstrap {
                obligations: retry.obligations,
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        for block in [500, 600] {
            assert!(
                h.tree.match_prefix(None, &[block]).workers().contains(&id),
                "live block {block} must survive the retry's abandonment",
            );
        }
        assert_eq!(h.cursors.lock().get(&id).copied(), Some(8));
        assert_eq!(rank_count(&tracker, "abandoned"), 1);
        assert_eq!(rank_count(&tracker, "gap"), 0);
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

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "snapshot state must be discarded for a gapped rank",
        );
        // Still held for the retry's graft, not replayed under it.
        assert_eq!(h.tree.match_prefix(None, &[42]).matched_blocks, 0);
    }

    /// With the retry already spent, a gap resolves cold: graft discarded, live
    /// deltas replayed, one `gap`.
    #[tokio::test]
    async fn a_second_gap_fails_the_rank_cold_and_replays_its_batches() {
        let id = worker_id("http://w1", 0);
        let tracker = pending_tracker(std::slice::from_ref(&id));
        // Spend the one retry up front.
        tracker.set(&id, BootstrapState::Failed);
        let retried = tracker
            .retry_after_gap(&id)
            .expect("first retry is granted");
        let mut h = spawn_pump_with_bootstrap(std::slice::from_ref(&id), tracker.clone());

        h.tx.send(WorkerEvent::Batch {
            worker: id.clone(),
            seq: 9,
            batch: batch(vec![stored(None, vec![42])]),
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: vec![retried],
                vetted: Box::new(vetted_for(&id, 5)),
            })
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Failed));
        assert!(h.bootstrap_rx.try_recv().is_err(), "no second retry");
        assert!(!h
            .tree
            .match_prefix(None, &[100, 200])
            .workers()
            .contains(&id));
        assert!(h.tree.match_prefix(None, &[42]).workers().contains(&id));
        assert_eq!(rank_count(&tracker, "gap"), 1);
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

    /// A probe verdict as the probe task posts it.
    fn splice_verdict(
        rank: &KvWorkerId,
        epoch: u64,
        watermark: i64,
        verdict: SpliceVerdict,
    ) -> PumpControl {
        PumpControl::SpliceProbe {
            rank: rank.clone(),
            epoch,
            watermark,
            verdict,
        }
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

    /// A peer that has applied a sequence ABOVE our watermark proves a batch we
    /// never received was published, so the grafted state goes.
    #[tokio::test]
    async fn pump_splice_probe_advanced_discards_grafted_state() {
        let id = worker_id("http://w1", 0);
        let (tracker, mut h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        h.ctrl_tx
            .send(splice_verdict(&id, epoch, 5, SpliceVerdict::Advanced))
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Pending));
        assert!(
            !h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "positive evidence of a missed batch must discard the graft",
        );
        assert!(
            h.cursors.lock().get(&id).is_none(),
            "cursor must be dropped"
        );
        let requeued = h.bootstrap_rx.try_recv().expect("gapped rank re-queued");
        assert_eq!(requeued.obligations[0].0, id);
        assert_eq!(requeued.late_join, LateJoin::Refused);
        assert_eq!(
            rank_count(&tracker, "gap"),
            0,
            "a retried gap is not a verdict; the retry's outcome is the rank's one count",
        );
        assert_eq!(rank_count(&tracker, "warm"), 0);
    }

    /// Silence with no witness of advancement is NOT evidence of a hole. Keeping
    /// the tree here is the whole reason the timeout probes instead of discarding:
    /// a quiet fleet would otherwise lose every warm tree on a timer.
    #[tokio::test]
    async fn pump_splice_probe_no_advance_keeps_grafted_state_and_counts_warm() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        h.ctrl_tx
            .send(splice_verdict(&id, epoch, 5, SpliceVerdict::NoAdvance))
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert!(
            h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "proof by absence must keep the grafted state",
        );
        assert_eq!(rank_count(&tracker, "warm"), 1);
    }

    /// An unanswerable probe resolves nothing: the rank stays warm AND still
    /// awaits proof, so a later witness can still demote it. Resolving `Unknown`
    /// either way would make an unreachable fleet decide the question.
    #[tokio::test]
    async fn pump_splice_probe_unknown_leaves_the_question_open() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        h.ctrl_tx
            .send(splice_verdict(&id, epoch, 5, SpliceVerdict::Unknown))
            .await
            .unwrap();
        // Still pending proof, so this second verdict must still be actionable.
        h.ctrl_tx
            .send(splice_verdict(&id, epoch, 5, SpliceVerdict::Advanced))
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "an Unknown verdict must not consume the pending proof; the later \
             Advanced verdict then gaps, which re-queues the rank",
        );
        assert_eq!(rank_count(&tracker, "warm"), 0);
    }

    /// A fleet that can never answer must not leave the rank probing forever: the
    /// graft is kept, but tallied under a label that says it was never witnessed
    /// rather than pretending it was proven.
    #[tokio::test]
    async fn pump_repeated_unknown_probes_resolve_as_unwitnessed() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        for _ in 0..MAX_UNKNOWN_PROBES {
            h.ctrl_tx
                .send(splice_verdict(&id, epoch, 5, SpliceVerdict::Unknown))
                .await
                .unwrap();
        }
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Recovered),
            "an unwitnessed graft is kept, not discarded",
        );
        assert!(h
            .tree
            .match_prefix(None, &[100, 200])
            .workers()
            .contains(&id));
        assert_eq!(rank_count(&tracker, "warm_unwitnessed"), 1);
        assert_eq!(
            rank_count(&tracker, "warm"),
            0,
            "unwitnessed must not be conflated with proven",
        );
    }

    /// A probe answer for a worker that was removed and re-added while it was in
    /// flight must not touch the new incarnation's state.
    #[tokio::test]
    async fn pump_splice_probe_from_a_stale_incarnation_is_ignored() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let stale_epoch = tracker.epoch_of(&id).expect("registered");
        tracker.forget(std::slice::from_ref(&id));
        tracker.register(std::slice::from_ref(&id));

        h.ctrl_tx
            .send(splice_verdict(&id, stale_epoch, 5, SpliceVerdict::Advanced))
            .await
            .unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(
            tracker.state_of(&id),
            Some(BootstrapState::Pending),
            "the re-registered incarnation must be left alone",
        );
        assert_eq!(rank_count(&tracker, "gap"), 0);
    }

    /// A gap retry re-grafts under the SAME epoch, so a verdict still in flight
    /// about the first graft's watermark (a second probe pass, say) must not
    /// demote the second graft: its question was about a different watermark.
    #[tokio::test]
    async fn pump_splice_probe_about_a_previous_graft_is_ignored() {
        let id = worker_id("http://w1", 0);
        let (tracker, h) = graft_with_deferred_proof(&id, 5).await;
        let epoch = tracker.epoch_of(&id).expect("registered");
        let advanced_past_5 = || splice_verdict(&id, epoch, 5, SpliceVerdict::Advanced);
        // First verdict gaps the graft and re-queues the rank as Pending.
        h.ctrl_tx.send(advanced_past_5()).await.unwrap();
        // The retry grafts a fresher snapshot, deferred again.
        h.ctrl_tx
            .send(PumpControl::ApplySnapshot {
                obligations: obligations(&tracker, std::slice::from_ref(&id)),
                vetted: Box::new(vetted_for(&id, 9)),
            })
            .await
            .unwrap();
        // A late duplicate about the FIRST graft's watermark.
        h.ctrl_tx.send(advanced_past_5()).await.unwrap();
        drop(h.tx);
        drop(h.ctrl_tx);
        h.pump.await.unwrap();

        assert_eq!(tracker.state_of(&id), Some(BootstrapState::Recovered));
        assert!(
            h.tree
                .match_prefix(None, &[100, 200])
                .workers()
                .contains(&id),
            "the retry's graft must survive a verdict about its predecessor",
        );
        assert_eq!(rank_count(&tracker, "gap"), 0, "the first gap was retried");
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
        // `remove_worker` forgets the tracker state before it sends `ForgetRanks`.
        tracker.forget(std::slice::from_ref(&id));
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
