// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Peer-snapshot bootstrap for the KV-event tree: the wire shape
//! ([`PeerSnapshot`]) and its producer constants, the fetch client
//! ([`fetch_snapshot`], [`fetch_cursors`]), the per-rank
//! [`BootstrapTracker`], and the [`PeerRegistry`] of peers a snapshot may be
//! pulled from.
//!
//! A freshly started replica subscribes to each worker's KV topic mid-stream:
//! ZMQ SUB delivers deltas from whatever sequence the publisher has reached,
//! so every block already resident in the engine's radix cache is invisible to
//! that replica. Until traffic re-stores those blocks the replica routes
//! cache-blind, and its own dispatches scatter prefixes across workers that
//! warm replicas were keeping consolidated. A tree snapshot pulled from a warm
//! sibling over HTTP, grafted beneath the live delta stream, closes that gap.
//!
//! # Why the body carries cursors and not just tree state
//!
//! A graft is only sound if the recipient can prove its own live stream picks
//! up exactly where the snapshot stops. The producer's per-rank last-applied
//! sequence is what makes that checkable: as a FILTER it says which buffered
//! deltas the snapshot already reflects, and as a WATERMARK it says the stream
//! must resume at `cursor + 1`. A hole there means a delta was lost, and a
//! lost `BlockRemoved` is a permanent false cache hit.
//!
//! The wire carries [`WireWorker`], never [`super::tree::KvWorkerId`], and
//! [`super::tree::HashTree::restore_snapshot`] stays module-internal, so
//! network input cannot become a `KvWorkerId` directly.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use flate2::read::GzDecoder;

use parking_lot::Mutex;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::tree::{HashTree, KvWorkerId, RestoreError, ShapeViolation, SnapshotNode};

/// Wire-format version. Bump on any incompatible change to [`PeerSnapshot`].
pub const SNAPSHOT_FORMAT: u32 = 1;

/// Path the producer serves snapshots on.
pub const SNAPSHOT_PATH: &str = "/internal/kv_snapshot";

/// Ceiling on a peer snapshot body, both as buffered off the wire and as
/// inflated, as a resource guard rather than a format bound.
///
/// The producing route is unauthenticated, so without it one response could
/// demand an arbitrarily large allocation — an identity body directly, a gzip
/// body from a small response — and an OOM kills a booting router before it
/// can log. A legitimate body is bounded by the fleet's total KV capacity in
/// blocks, orders of magnitude below this. Generous on purpose: rejecting a
/// real snapshot costs a cold boot, so this should only ever catch a body no
/// honest peer would send.
const MAX_INFLATED_SNAPSHOT_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Query parameter by which a caller states how stale a cached snapshot it
/// will accept, in milliseconds. See [`PRODUCER_CACHE_TTL`].
pub const MAX_AGE_PARAM: &str = "max_age_ms";

/// Query parameter by which a caller asks for the cursor table alone, with
/// no tree. See [`fetch_cursors`].
pub const CURSORS_ONLY_PARAM: &str = "cursors_only";

/// Default reuse window for callers that send no [`MAX_AGE_PARAM`]. A cached
/// export predates the caller's subscription, and events in that gap reach
/// neither side, so callers that care send `max_age_ms` and the producer
/// rebuilds.
pub const PRODUCER_CACHE_TTL: Duration = Duration::from_secs(2);

/// Default upper bound on one snapshot fetch that is making progress. Stalls
/// are caught by [`SNAPSHOT_FETCH_READ_TIMEOUT`] and dead peers by
/// [`SNAPSHOT_FETCH_CONNECT_TIMEOUT`], so this is sized for a large healthy
/// body (tens of MB gzipped, tens of seconds).
pub const DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP: Duration = Duration::from_secs(120);

/// Bound on establishing the connection to a peer. A peer that is gone fails
/// this fast, so a caller reaches its next candidate without spending the
/// per-fetch budget on a dead one.
pub const SNAPSHOT_FETCH_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Bound on the gap BETWEEN body chunks, not on the whole response.
///
/// This is what separates "hung" from "big": a total-request timeout fires
/// identically on a peer that sent nothing and on one streaming a large body
/// steadily. Bounding idle time lets an arbitrarily large healthy snapshot
/// complete while still cutting a stalled peer loose in seconds.
pub const SNAPSHOT_FETCH_READ_TIMEOUT: Duration = Duration::from_secs(15);

/// Per-rank bootstrap state. `settled()` waits on `Pending` ranks until the
/// deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapState {
    /// Registered; no snapshot applied yet.
    Pending,
    /// Snapshot grafted and cursor seeded.
    Recovered,
    /// No usable snapshot; the rank runs on live deltas only.
    Failed,
}

impl BootstrapState {
    /// Numeric encoding for the `sgl_router_kv_bootstrap_state` gauge.
    pub fn as_metric(self) -> u64 {
        match self {
            Self::Pending => 0,
            Self::Recovered => 1,
            Self::Failed => 2,
        }
    }

    pub fn is_terminal(self) -> bool {
        !matches!(self, Self::Pending)
    }
}

/// How one fetch from one peer turned out; counted per peer attempt, not per
/// rank (see [`RankOutcome`]). A closed set so no call site can mint a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotOutcome {
    /// Usable body.
    Accepted,
    /// Peer did not answer, or answered non-200 (including the 404 an older
    /// router image returns for an endpoint it does not serve).
    Unreachable,
    /// Answered, but holds no state to graft for the ranks being bootstrapped.
    /// That includes WARM peers whose tree shares no carriers with our live
    /// workers or tracks none of those ranks, so this is not
    /// [`PeerSnapshot::is_cold`] and does not count toward a cold-fleet verdict.
    ColdPeer,
    /// Answered with an untrustworthy body: any [`VetError`] whose
    /// [`VetError::outcome`] is `Rejected`.
    Rejected,
}

impl SnapshotOutcome {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Unreachable => "unreachable",
            Self::ColdPeer => "cold_peer",
            Self::Rejected => "rejected",
        }
    }
}

/// How one bounded peer sweep ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepOutcome {
    /// A peer supplied a usable snapshot.
    Found,
    /// Discovery confirmed there are no siblings.
    NoPeers,
    /// Every sibling proved it holds no state.
    FleetCold,
    /// The deadline expired first.
    TimedOut,
}

impl SweepOutcome {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Found => "found",
            Self::NoPeers => "no_peers",
            Self::FleetCold => "fleet_cold",
            Self::TimedOut => "timed_out",
        }
    }
}

/// How one rank's bootstrap ended; recorded once per rank per incarnation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankOutcome {
    /// Grafted, and the splice with the live stream proven.
    Warm,
    /// The live stream did not join the snapshot's watermark; graft discarded.
    Gap,
    /// Grafted and kept, but no batch or peer witnessed the splice.
    WarmUnwitnessed,
    /// The accepted snapshot carried no cursor for this rank.
    Uncovered,
    /// No peer supplied a usable snapshot.
    Abandoned,
    /// Held batches hit their cap before a snapshot arrived; bootstrap was
    /// abandoned and they were replayed as live deltas.
    Overflow,
    /// The publisher restarted mid-bootstrap.
    PublisherReset,
    /// The tree refused the snapshot's structure.
    TreeRejected,
}

impl RankOutcome {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Warm => "warm",
            Self::Gap => "gap",
            Self::WarmUnwitnessed => "warm_unwitnessed",
            Self::Uncovered => "uncovered",
            Self::Abandoned => "abandoned",
            Self::Overflow => "overflow",
            Self::PublisherReset => "publisher_reset",
            Self::TreeRejected => "tree_rejected",
        }
    }
}

/// A worker identity as it appears on the snapshot wire. Not a
/// [`KvWorkerId`]; see the module docs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireWorker {
    pub url: String,
    pub dp_rank: u32,
}

/// Outbound only: describing a local id on the wire mints nothing. No
/// conversion exists the other way.
impl From<&KvWorkerId> for WireWorker {
    fn from(w: &KvWorkerId) -> Self {
        Self {
            url: w.url.clone(),
            dp_rank: w.dp_rank,
        }
    }
}

/// A peer replica's view of the KV tree, plus the cursors needed to splice it
/// under a live delta stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeerSnapshot {
    /// See [`SNAPSHOT_FORMAT`].
    pub format: u32,
    /// Producer's established block size. Block hashes are only comparable at
    /// the same page size, so a mismatch is fatal to the snapshot.
    pub block_size: u32,
    /// Producer's hashing mode (EAGLE-family workers hash token bigrams).
    pub is_bigram: bool,
    /// True when the producer's own bootstrap has settled, its hash config is
    /// established, and its tree holds nodes, so a cold, still-bootstrapping
    /// or half-configured replica is never copied.
    pub producer_ready: bool,
    /// Worker table; node carrier lists index into this.
    pub workers: Vec<WireWorker>,
    /// `(worker-table index, last-applied seq)` at export time; the stream
    /// resumes at `cursor + 1`.
    pub cursors: Vec<(u32, i64)>,
    /// Tree nodes in dependency order; see [`SnapshotNode`].
    pub nodes: Vec<SnapshotNode>,
}

impl PeerSnapshot {
    /// Not ready, or empty tree.
    pub fn is_cold(&self) -> bool {
        !self.producer_ready || self.holds_no_state()
    }

    /// Empty tree. Weaker than [`Self::is_cold`]: a peer mid-bootstrap reports
    /// `producer_ready: false` with nodes.
    pub fn holds_no_state(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Last-applied seq the producer reports for `(url, dp_rank)`. Sequence
    /// numbers are the publisher's, so any observer's value is evidence of
    /// publisher progress.
    pub fn wire_cursor_for(&self, url: &str, dp_rank: u32) -> Option<i64> {
        let idx = self
            .workers
            .iter()
            .position(|w| w.url == url && w.dp_rank == dp_rank)? as u32;
        self.cursors
            .iter()
            .find(|(i, _)| *i == idx)
            .map(|(_, seq)| *seq)
    }
}

/// Why a wire snapshot was refused before any tree mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VetError {
    UnknownFormat {
        got: u32,
        want: u32,
    },
    /// A node's `parent` was not a backward reference into the node list.
    InvalidParentReference {
        index: usize,
        parent: u32,
    },
    /// A node's `tiers` was neither empty nor the same length as its
    /// `workers`, so carriers cannot be paired with their tiers.
    /// [`SnapshotNode::retain_carriers`] drops a carrier that lacks a tier
    /// entry, so this check keeps malformed pairing from being silently
    /// dropped.
    TierTableMismatch {
        index: usize,
    },
    /// Two cursor entries resolve to the same worker, so its watermark is
    /// ambiguous and [`VettedSnapshot::cursor_for`] would silently pick one.
    DuplicateCursor {
        worker: u32,
    },
    /// Two worker-table entries name the same live worker. An honest export
    /// lists each worker once, and merging the entries would leave nodes that
    /// name one carrier twice.
    DuplicateWorker {
        index: usize,
    },
    BlockSizeMismatch {
        peer: u32,
        local: u32,
    },
    ProducerCold,
    /// The snapshot was well formed, but nothing in it survives vetting for this
    /// replica — every carrier was a worker we do not know, or held the node
    /// only on tiers this build does not rank. Unlike `ProducerCold`, the peer
    /// has a real tree with no overlap with ours.
    ///
    /// Refused rather than accepted as empty: a seeded cursor with no tree
    /// behind it filters away every live delta at or below its watermark.
    NothingUsable {
        wire_nodes: usize,
        dropped_workers: usize,
    },
}

impl std::fmt::Display for VetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownFormat { got, want } => {
                write!(f, "unknown snapshot format {got} (want {want})")
            }
            Self::InvalidParentReference { index, parent } => write!(
                f,
                "snapshot node {index} has parent {parent}, not a backward reference",
            ),
            Self::TierTableMismatch { index } => write!(
                f,
                "snapshot node {index} has a tiers list that does not pair with its workers",
            ),
            Self::DuplicateCursor { worker } => write!(
                f,
                "snapshot has more than one cursor for worker {worker}; the watermark is ambiguous",
            ),
            Self::DuplicateWorker { index } => write!(
                f,
                "snapshot worker {index} repeats an earlier worker-table entry",
            ),
            Self::BlockSizeMismatch { peer, local } => write!(
                f,
                "peer block_size {peer} disagrees with local {local}; block hashes are incomparable",
            ),
            Self::NothingUsable {
                wire_nodes,
                dropped_workers,
            } => write!(
                f,
                "none of the peer's {wire_nodes} nodes survive vetting \
                 ({dropped_workers} of its workers are unknown here)",
            ),
            Self::ProducerCold => write!(
                f,
                "peer is not a usable source (still bootstrapping, or settled with an empty tree)",
            ),
        }
    }
}

impl VetError {
    pub fn outcome(&self) -> SnapshotOutcome {
        match self {
            // Not the peer's fault and not permanent — it may discover our
            // workers moments later, so this must stay retriable.
            Self::NothingUsable { .. } | Self::ProducerCold => SnapshotOutcome::ColdPeer,
            _ => SnapshotOutcome::Rejected,
        }
    }
}

/// A snapshot whose worker identities have been resolved against the local
/// live set and whose carrier lists have been remapped onto the surviving
/// workers.
///
/// Only [`VettedSnapshot::from_wire`] builds one, and
/// [`VettedSnapshot::graft_into`] is the only route from it to the tree's
/// restore path; the module docs describe the trust boundary this upholds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VettedSnapshot {
    /// Locally-minted worker ids, resolved from the live set.
    /// [`Self::retain_workers`] filters carriers and cursors, never this
    /// table.
    worker_table: Vec<KvWorkerId>,
    /// Nodes with carrier indices remapped into `worker_table`, pruned by
    /// [`VettedSnapshot::prune_carrier_less`].
    nodes: Vec<SnapshotNode>,
    /// `(worker, last-applied seq)` for workers that survived vetting.
    cursors: Vec<(KvWorkerId, i64)>,
    /// Wire workers the local replica does not know. Expected and benign
    /// during a rolling worker change; logged, and worth watching if
    /// persistently large.
    dropped_workers: usize,
}

impl VettedSnapshot {
    /// Nodes that survived vetting, for logging.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Distinct workers that survived vetting.
    pub fn worker_count(&self) -> usize {
        self.worker_table.len()
    }

    /// Whether `id` survived vetting.
    pub fn has_worker(&self, id: &KvWorkerId) -> bool {
        self.worker_table.contains(id)
    }

    /// Wire workers this replica does not know; see the field docs.
    pub fn dropped_workers(&self) -> usize {
        self.dropped_workers
    }

    /// `(carried, structure)`: nodes with at least one surviving carrier vs
    /// carrier-less interior kept on a path to a carried descendant. Structure
    /// nodes cannot answer a query but can turn one into a hit-shaped miss, so
    /// their count is surfaced rather than buried inside `node_count`.
    pub(crate) fn carrier_counts(&self) -> (usize, usize) {
        let structure = self.nodes.iter().filter(|n| n.workers.is_empty()).count();
        (self.nodes.len() - structure, structure)
    }

    /// Graft this snapshot's nodes into `tree`, returning how many were applied.
    ///
    /// Hands over the ids vetting resolved against the live set, the
    /// precondition the restore path trusts. Call it from the tree's single
    /// writer, like every tree mutation.
    pub fn graft_into(&self, tree: &HashTree) -> Result<usize, RestoreError> {
        tree.restore_snapshot(&self.worker_table, &self.nodes)
    }

    /// Build one directly, for tests that need a specific shape without a
    /// wire round trip. Test-only because it skips vetting.
    #[cfg(test)]
    pub fn from_parts_for_test(
        worker_table: Vec<KvWorkerId>,
        nodes: Vec<SnapshotNode>,
        cursors: Vec<(KvWorkerId, i64)>,
        dropped_workers: usize,
    ) -> Self {
        Self {
            worker_table,
            nodes,
            cursors,
            dropped_workers,
        }
    }

    /// Drop every carrier and cursor whose worker is not in `keep`, then prune
    /// the structure that strands. Lets a caller graft for a subset of the
    /// vetted workers.
    pub fn retain_workers(&mut self, keep: &HashSet<KvWorkerId>) {
        let allowed: Vec<bool> = self.worker_table.iter().map(|w| keep.contains(w)).collect();
        if allowed.iter().all(|&a| a) {
            return;
        }
        let is_allowed = |w: u32| allowed.get(w as usize) == Some(&true);
        for node in &mut self.nodes {
            if node.workers.iter().all(|&w| is_allowed(w)) {
                continue;
            }
            node.retain_carriers(|w, _| is_allowed(w).then_some(w));
        }
        self.cursors.retain(|(w, _)| keep.contains(w));
        let pruned = self.prune_carrier_less();
        if pruned > 0 {
            debug!(
                pruned,
                remaining = self.nodes.len(),
                "kv-bootstrap: dropped carrier-less nodes after filtering to pending ranks",
            );
        }
    }

    /// Drop nodes that carry no worker and have no surviving descendant that
    /// does, remapping parent indices. Returns the number of nodes dropped.
    ///
    /// `match_prefix` reports the carriers of the DEEPEST matched node, so a
    /// carrier-less node grafted below a carrying one turns a real cache hit
    /// into a match with no workers, and nothing ever reclaims it. Vetting
    /// leaves such nodes routinely: carriers of workers this replica has not
    /// discovered, carriers held only on tiers this build does not rank, and
    /// carriers [`Self::retain_workers`] filters out.
    pub fn prune_carrier_less(&mut self) -> usize {
        let n = self.nodes.len();
        // Parents precede children, so one reverse pass settles every node,
        // and a kept node's parent is always kept.
        let mut keep = vec![false; n];
        for (i, node) in self.nodes.iter().enumerate().rev() {
            keep[i] |= !node.workers.is_empty();
            if let (true, Some(p)) = (keep[i], node.parent) {
                keep[p as usize] = true;
            }
        }
        let pruned = keep.iter().filter(|&&k| !k).count();
        if pruned == 0 {
            return 0;
        }

        const DROPPED: u32 = u32::MAX;
        let mut new_index = vec![DROPPED; n];
        let mut out: Vec<SnapshotNode> = Vec::with_capacity(n - pruned);
        let nodes = std::mem::take(&mut self.nodes);
        for (i, (mut node, kept)) in nodes.into_iter().zip(keep).enumerate() {
            if !kept {
                continue;
            }
            if let Some(p) = node.parent {
                let np = new_index[p as usize];
                debug_assert_ne!(np, DROPPED, "node {i} is kept but its parent {p} is not");
                if np == DROPPED {
                    continue;
                }
                node.parent = Some(np);
            }
            new_index[i] = out.len() as u32;
            out.push(node);
        }
        self.nodes = out;
        n - self.nodes.len()
    }

    /// Whether the snapshot has a cursor for any of `ranks`; one without
    /// cannot seed a watermark for them.
    pub fn covers_any(&self, ranks: &[KvWorkerId]) -> bool {
        ranks.iter().any(|r| self.cursor_for(r).is_some())
    }

    /// Last-applied sequence the producer had for `worker`, or `None` when it
    /// did not track that worker.
    pub fn cursor_for(&self, worker: &KvWorkerId) -> Option<i64> {
        self.cursors
            .iter()
            .find(|(w, _)| w == worker)
            .map(|(_, seq)| *seq)
    }

    /// Validate a wire snapshot and resolve its worker table against `live`.
    ///
    /// `local_block_size` is `None` before any worker has established one, in
    /// which case the peer's value is accepted — there is nothing yet to
    /// contradict it, and `add_worker` will reject any worker that disagrees.
    ///
    /// The producer's `is_bigram` stamp is not vetted: hashing mode belongs to
    /// the publishing worker, and every surviving node is carried by a worker
    /// this replica discovered and hashes for in its own established mode.
    pub fn from_wire(
        snap: PeerSnapshot,
        live: &HashSet<KvWorkerId>,
        local_block_size: Option<u32>,
    ) -> Result<Self, VetError> {
        if snap.format != SNAPSHOT_FORMAT {
            return Err(VetError::UnknownFormat {
                got: snap.format,
                want: SNAPSHOT_FORMAT,
            });
        }
        // Refuse a producer that reports not-ready, or whose tree is empty
        // whatever it reports.
        if snap.is_cold() {
            return Err(VetError::ProducerCold);
        }
        if let Some(local) = local_block_size {
            if local != snap.block_size {
                return Err(VetError::BlockSizeMismatch {
                    peer: snap.block_size,
                    local,
                });
            }
        }

        // `prune_carrier_less` indexes by `parent`, so bound it and pair tiers
        // before anything reads the nodes.
        for (i, rec) in snap.nodes.iter().enumerate() {
            rec.check_shape(i).map_err(|v| match v {
                ShapeViolation::NonBackwardParent { parent } => {
                    VetError::InvalidParentReference { index: i, parent }
                }
                ShapeViolation::TierTableMismatch => VetError::TierTableMismatch { index: i },
            })?;
        }

        // Resolve wire identities to live ones. `remap[i]` is the new index of
        // wire worker `i`, or `None` when this replica does not know it.
        let mut worker_table: Vec<KvWorkerId> = Vec::new();
        let mut seen: HashSet<&KvWorkerId> = HashSet::new();
        let mut remap: Vec<Option<u32>> = Vec::with_capacity(snap.workers.len());
        let mut dropped_workers = 0usize;
        for (index, w) in snap.workers.iter().enumerate() {
            // Construct only to look up; the id kept is the one from `live`,
            // preserving registry provenance.
            let probe = KvWorkerId::new(w.url.clone(), w.dp_rank);
            match live.get(&probe) {
                Some(known) => {
                    if !seen.insert(known) {
                        return Err(VetError::DuplicateWorker { index });
                    }
                    remap.push(Some(worker_table.len() as u32));
                    worker_table.push(known.clone());
                }
                None => {
                    remap.push(None);
                    dropped_workers += 1;
                }
            }
        }
        let resolve = |w: u32| remap.get(w as usize).copied().flatten();

        // Drop carriers of unknown workers and carriers on no ranked tier, so
        // `prune_carrier_less` sees every node the restore path would leave
        // carrier-less.
        let nodes: Vec<SnapshotNode> = snap
            .nodes
            .into_iter()
            .map(|mut n| {
                // Rebuilding both lists costs two allocations per node, and on
                // the common path — every wire worker known, every tier ranked —
                // nothing changes. Only pay for the nodes it actually changes.
                let changes = n
                    .workers
                    .iter()
                    .enumerate()
                    .any(|(k, &w)| resolve(w) != Some(w) || n.carrier_tiers(k).is_empty());
                if changes {
                    n.retain_carriers(|w, tiers| if tiers.is_empty() { None } else { resolve(w) });
                }
                n
            })
            .collect();

        let mut cursors: Vec<(KvWorkerId, i64)> = Vec::with_capacity(snap.cursors.len());
        let mut cursor_seen = vec![false; worker_table.len()];
        for (idx, seq) in snap.cursors {
            let Some(slot) = resolve(idx) else {
                continue;
            };
            if std::mem::replace(&mut cursor_seen[slot as usize], true) {
                return Err(VetError::DuplicateCursor { worker: idx });
            }
            cursors.push((worker_table[slot as usize].clone(), seq));
        }

        let wire_nodes = nodes.len();
        let mut vetted = Self {
            worker_table,
            nodes,
            cursors,
            dropped_workers,
        };
        let pruned = vetted.prune_carrier_less();
        if pruned > 0 {
            debug!(
                pruned,
                remaining = vetted.nodes.len(),
                dropped_workers,
                "kv-bootstrap: dropped carrier-less nodes left by unknown workers",
            );
        }
        // Pruning can empty a node list the cold gate saw non-empty; see
        // `VetError::NothingUsable`.
        if vetted.nodes.is_empty() {
            return Err(VetError::NothingUsable {
                wire_nodes,
                dropped_workers,
            });
        }
        Ok(vetted)
    }
}

/// Rate-limit key for the snapshot-attempt log: one entry per peer per
/// verdict.
type PeerOutcomeKey = (String, &'static str);

/// Rate-limit state for one [`PeerOutcomeKey`]. The last-seen detail is kept
/// so a cause CHANGE under the same verdict is surfaced rather than throttled
/// away.
#[derive(Debug, Default)]
struct AttemptLog {
    attempts: u64,
    last_detail: String,
}

/// A repeated verdict against one peer is logged at info on every this-many
/// attempts, and at debug otherwise.
const ATTEMPT_LOG_EVERY: u64 = 100;

/// Counts keyed by a fixed `&'static str` label set, so cardinality is
/// bounded. Sampled by the metrics surface at scrape time.
#[derive(Debug, Default)]
struct LabelTally(Mutex<HashMap<&'static str, u64>>);

impl LabelTally {
    fn bump(&self, label: &'static str) {
        *self.0.lock().entry(label).or_insert(0) += 1;
    }

    fn snapshot(&self) -> Vec<(&'static str, u64)> {
        self.0.lock().iter().map(|(k, v)| (*k, *v)).collect()
    }
}

/// Per-rank bootstrap progress and whether initial bootstrap has settled.
/// Once settled, `settled()` stays true, so a rank registered later
/// (scale-up) cannot un-settle it.
#[derive(Debug)]
pub struct BootstrapTracker {
    states: Mutex<HashMap<KvWorkerId, BootstrapState>>,
    deadline: Mutex<Option<Instant>>,
    timeout: Duration,
    /// Upper bound on one peer-snapshot fetch; defaults to
    /// [`DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP`].
    fetch_cap: Duration,
    /// Whether peer bootstrap is configured. Not derivable from `settled()`:
    /// a disabled tracker and a finished one both report settled.
    enabled: bool,
    /// Set once `settled()` first answers true AFTER the first registration.
    /// An expiry observed before any rank is registered answers true without
    /// latching, so the re-arm at first registration still gates readiness.
    latched: AtomicBool,
    /// Whether the one-shot re-arm at first worker discovery has happened.
    ///
    /// Keyed on this rather than on `states.is_empty()`: the map empties again
    /// whenever `forget` removes the last rank, so an engine flapping
    /// remove/re-add would re-arm on every cycle and hold the readiness gate for
    /// an unbounded multiple of the configured timeout.
    rearmed: AtomicBool,
    /// Per-rank incarnation, minted afresh when a rank is registered after
    /// `forget`, so a result started for an earlier incarnation is
    /// recognisably stale.
    epochs: Mutex<HashMap<KvWorkerId, u64>>,
    epoch_seq: AtomicU64,
    /// Ranks already given their one post-gap retry this incarnation; `forget`
    /// clears the mark.
    gap_retried: Mutex<HashSet<KvWorkerId>>,
    /// Keyed by [`SnapshotOutcome::as_label`].
    peer_outcomes: LabelTally,
    /// Keyed by [`RankOutcome::as_label`].
    rank_outcomes: LabelTally,
    /// Rate-limit state for the attempt log in [`Self::record_peer_outcome`].
    peer_attempts: Mutex<HashMap<PeerOutcomeKey, AttemptLog>>,
    /// Keyed by [`SweepOutcome::as_label`].
    sweep_results: LabelTally,
}

/// `now + timeout`, clamped instead of panicking: `Instant + Duration` panics
/// on overflow, and an absurd timeout means "effectively never", not "abort".
fn deadline_after(timeout: Duration) -> Instant {
    const FAR_FUTURE: Duration = Duration::from_secs(100 * 365 * 24 * 60 * 60);
    let now = Instant::now();
    now.checked_add(timeout)
        .or_else(|| now.checked_add(FAR_FUTURE))
        .unwrap_or(now)
}

impl BootstrapTracker {
    pub fn new(timeout: Duration) -> Self {
        Self::new_with_fetch_cap(timeout, DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP)
    }

    pub fn new_with_fetch_cap(timeout: Duration, fetch_cap: Duration) -> Self {
        Self {
            fetch_cap,
            states: Mutex::new(HashMap::new()),
            // Armed at construction so `settled()` turns true by the deadline
            // even if no rank is ever registered; `register` re-arms once so the
            // budget normally runs from first worker discovery.
            deadline: Mutex::new(Some(deadline_after(timeout))),
            timeout,
            enabled: true,
            latched: AtomicBool::new(false),
            rearmed: AtomicBool::new(false),
            epochs: Mutex::new(HashMap::new()),
            epoch_seq: AtomicU64::new(0),
            gap_retried: Mutex::new(HashSet::new()),
            peer_outcomes: LabelTally::default(),
            rank_outcomes: LabelTally::default(),
            peer_attempts: Mutex::new(HashMap::new()),
            sweep_results: LabelTally::default(),
        }
    }

    /// Tally one snapshot FETCH against one peer, and log it.
    pub fn record_peer_outcome(&self, outcome: SnapshotOutcome, peer: &str, detail: Option<&str>) {
        self.peer_outcomes.bump(outcome.as_label());
        if outcome == SnapshotOutcome::Accepted {
            return;
        }
        let detail = detail.unwrap_or("");
        let (attempts, detail_changed) = {
            let mut logs = self.peer_attempts.lock();
            let log = logs
                .entry((peer.to_string(), outcome.as_label()))
                .or_default();
            log.attempts += 1;
            let changed = log.last_detail != detail;
            if changed {
                log.last_detail = detail.to_string();
            }
            (log.attempts, changed && log.attempts > 1)
        };
        // The first failure against a peer is signal; a repeat of the same
        // verdict is not — but repeated attempts against unreachable or
        // non-covering peers must still show progress at the default log level.
        // A detail CHANGE under the same verdict (connection-refused becoming
        // DNS, 404 becoming 503) is a new cause, not a repeat: surface it too.
        let surface = attempts == 1 || attempts % ATTEMPT_LOG_EVERY == 0 || detail_changed;
        macro_rules! log_attempt {
            ($level:ident) => {
                $level!(
                    peer = %peer,
                    outcome = outcome.as_label(),
                    attempts,
                    detail,
                    "kv-bootstrap: snapshot attempt did not yield state",
                )
            };
        }
        if surface {
            log_attempt!(info);
        } else {
            log_attempt!(debug);
        }
    }

    /// Tally one rank's final verdict; record each rank once.
    pub fn record_rank_outcome(&self, outcome: RankOutcome) {
        self.rank_outcomes.bump(outcome.as_label());
    }

    /// Per-peer-attempt tallies for the metrics surface.
    pub fn peer_outcome_counts(&self) -> Vec<(&'static str, u64)> {
        self.peer_outcomes.snapshot()
    }

    /// Per-rank tallies for the metrics surface.
    pub fn rank_outcome_counts(&self) -> Vec<(&'static str, u64)> {
        self.rank_outcomes.snapshot()
    }

    /// Tally one sweep's terminal verdict. `peers_tried` is accepted for
    /// callers that gate on it and is not stored.
    pub fn record_sweep_result(&self, result: SweepOutcome, peers_tried: usize) {
        let _ = peers_tried;
        self.sweep_results.bump(result.as_label());
    }

    /// Per-sweep-verdict tallies for the metrics surface.
    pub fn sweep_result_counts(&self) -> Vec<(&'static str, u64)> {
        self.sweep_results.snapshot()
    }

    /// A tracker that is settled from the start, for the paths where peer
    /// bootstrap is disabled entirely.
    pub fn disabled() -> Self {
        let mut t = Self::new(Duration::ZERO);
        t.enabled = false;
        t.latched.store(true, Ordering::Relaxed);
        t
    }

    /// Whether peer bootstrap is configured. Unlike [`BootstrapTracker::settled`]
    /// this never flips: it answers "may ranks bootstrap at all?", not "is
    /// readiness still waiting?".
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Upper bound on one peer-snapshot fetch.
    pub fn fetch_cap(&self) -> Duration {
        self.fetch_cap
    }

    /// Register ranks as `Pending` and return obligations for the ones newly
    /// inserted: each paired with a freshly minted incarnation number a later
    /// control message must still match to be allowed to act on it.
    ///
    /// A rank already registered keeps its state and incarnation and yields no
    /// obligation: whoever holds its existing obligation owns it. Decided under
    /// the `states` lock, so concurrent callers registering the same rank get
    /// exactly one obligation between them. A rank [`Self::forget`] removed is
    /// new again and gets a new incarnation.
    ///
    /// Ranks added after the tracker has latched are recorded (so metrics stay
    /// accurate) but cannot un-settle readiness. The first registration is not
    /// such a case: nothing latches before it (see [`Self::settled`]), so the
    /// re-arm below is what the readiness gate measures from.
    pub fn register(&self, ids: &[KvWorkerId]) -> Vec<(KvWorkerId, u64)> {
        if ids.is_empty() {
            return Vec::new();
        }
        let mut states = self.states.lock();
        // Re-arm exactly once, at first worker discovery, so the budget is not
        // spent by slow discovery. Strictly one-shot — see `rearmed`. Done
        // under the `states` lock, which `settled()` also holds while it
        // decides whether to latch, so the two cannot interleave.
        if !self.rearmed.swap(true, Ordering::Relaxed) {
            *self.deadline.lock() = Some(deadline_after(self.timeout));
        }
        let mut epochs = self.epochs.lock();
        let mut obligations = Vec::with_capacity(ids.len());
        for id in ids {
            if states.contains_key(id) {
                continue;
            }
            states.insert(id.clone(), BootstrapState::Pending);
            let epoch = self.epoch_seq.fetch_add(1, Ordering::Relaxed) + 1;
            epochs.insert(id.clone(), epoch);
            obligations.push((id.clone(), epoch));
        }
        obligations
    }

    /// Current incarnation of `id`, or `None` if it is not registered.
    pub fn epoch_of(&self, id: &KvWorkerId) -> Option<u64> {
        self.epochs.lock().get(id).copied()
    }

    /// Update a REGISTERED rank's state; an unregistered rank is ignored.
    ///
    /// Non-creating because `forget` is the authority on membership: an
    /// inserting `set` could resurrect a forgotten rank's state without an
    /// epoch, and `register` would then keep that stale state for the re-added
    /// rank instead of starting it `Pending`.
    pub fn set(&self, id: &KvWorkerId, state: BootstrapState) {
        if let Some(slot) = self.states.lock().get_mut(id) {
            *slot = state;
        }
    }

    pub fn forget(&self, ids: &[KvWorkerId]) {
        // Lock order: states, epochs, gap_retried — `retry_after_gap` takes
        // the same order.
        let mut states = self.states.lock();
        let mut epochs = self.epochs.lock();
        let mut gap_retried = self.gap_retried.lock();
        for id in ids {
            states.remove(id);
            epochs.remove(id);
            gap_retried.remove(id);
        }
    }

    /// Move a gap-discarded rank back to `Pending` so it can be swept again,
    /// returning the obligation to queue.
    ///
    /// One retry per incarnation; `forget` clears the mark. `None` when this
    /// incarnation already retried, the rank was forgotten, or it is not
    /// `Failed`. The caller must queue a returned obligation, since nothing
    /// else resolves the `Pending` state it leaves. A latched `settled()` is
    /// unaffected.
    pub fn retry_after_gap(&self, id: &KvWorkerId) -> Option<(KvWorkerId, u64)> {
        let mut states = self.states.lock();
        if states.get(id) != Some(&BootstrapState::Failed) {
            return None;
        }
        // Same lock order as `register` (states, then epochs).
        let epoch = *self.epochs.lock().get(id)?;
        if !self.gap_retried.lock().insert(id.clone()) {
            return None;
        }
        states.insert(id.clone(), BootstrapState::Pending);
        Some((id.clone(), epoch))
    }

    pub fn state_of(&self, id: &KvWorkerId) -> Option<BootstrapState> {
        self.states.lock().get(id).copied()
    }

    /// Snapshot of every tracked rank, for the metrics surface.
    pub fn states(&self) -> Vec<(KvWorkerId, BootstrapState)> {
        self.states
            .lock()
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Whether every rank a `Pending` entry exists for has reached a terminal
    /// state, or the deadline has passed.
    ///
    /// Latches on the first `true` observed after the first registration. An
    /// expiry seen before any rank is registered answers true WITHOUT
    /// latching, so the first registration's re-arm still governs.
    ///
    /// The whole decision runs under the `states` lock, which `register` holds
    /// while it re-arms, so a latch can never be computed from the
    /// construction-time deadline and stored after the re-arm.
    ///
    /// The deadline is always armed (see [`BootstrapTracker::new`]), so this
    /// cannot stay `false` forever regardless of what does or does not get
    /// registered.
    pub fn settled(&self) -> bool {
        if self.latched.load(Ordering::Relaxed) {
            return true;
        }
        let pending = {
            let states = self.states.lock();
            let pending = states.values().filter(|s| !s.is_terminal()).count();
            let all_terminal = !states.is_empty() && pending == 0;
            let expired = self.deadline.lock().is_some_and(|d| Instant::now() >= d);
            if !(all_terminal || expired) {
                return false;
            }
            if !self.rearmed.load(Ordering::Relaxed) {
                return true;
            }
            self.latched.store(true, Ordering::Relaxed);
            pending
        };
        if pending > 0 {
            warn!(
                timeout_ms = self.timeout.as_millis(),
                pending,
                "kv-bootstrap: deadline elapsed with ranks still pending; \
                 serving with a partially warmed cache-aware tree",
            );
        }
        true
    }

    /// Time left before the deadline forces settlement, or `None` when no
    /// deadline is armed yet.
    pub fn time_remaining(&self) -> Option<Duration> {
        self.deadline
            .lock()
            .map(|d| d.saturating_duration_since(Instant::now()))
    }

    /// Move the deadline to now, so a test can observe expiry without
    /// sleeping against a wall-clock budget.
    #[cfg(test)]
    fn expire_deadline_for_test(&self) {
        *self.deadline.lock() = Some(Instant::now());
    }
}

/// Sibling replicas a snapshot may be pulled from. An empty set is conclusive
/// only after a sync and only if siblings were never seen: before the first
/// delivery it means "not told yet", and after siblings were seen it is a
/// transient dip (slice repack, rolling update). All state is behind one lock
/// so a reader never pairs fields from different writes.
#[derive(Debug, Default)]
pub struct PeerRegistry {
    state: Mutex<PeerState>,
}

#[derive(Debug, Default)]
struct PeerState {
    peers: Vec<String>,
    /// Whether a peer set has been delivered at least once, even an empty one.
    synced: bool,
    /// Whether a non-empty peer set has ever been delivered.
    ever_had_peers: bool,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the peer set wholesale, as a relist or a slice event produces.
    pub fn replace(&self, peers: Vec<String>) {
        let changed = {
            let mut state = self.state.lock();
            state.synced = true;
            state.ever_had_peers |= !peers.is_empty();
            let changed = (state.peers != peers).then(|| peers.clone());
            state.peers = peers;
            changed
        };
        if let Some(peers) = changed {
            info!(count = peers.len(), peers = ?peers, "kv-bootstrap: peer set updated");
        }
    }

    pub fn len(&self) -> usize {
        self.state.lock().peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.lock().peers.is_empty()
    }

    /// Whether peer discovery has reported at least once. An empty peer set is
    /// only conclusive once this is true.
    pub fn synced(&self) -> bool {
        self.state.lock().synced
    }

    /// Peer count and `synced`, read together.
    pub fn len_and_synced(&self) -> (usize, bool) {
        let state = self.state.lock();
        (state.peers.len(), state.synced)
    }

    /// True when discovery has confirmed this replica has no siblings, so
    /// there is no point waiting for one.
    pub fn known_to_have_no_peers(&self) -> bool {
        let state = self.state.lock();
        state.synced && state.peers.is_empty() && !state.ever_had_peers
    }

    /// Candidate peers in shuffled order, so simultaneous boots spread their
    /// snapshot fetches instead of stampeding whichever peer sorts first.
    pub fn candidates(&self) -> Vec<String> {
        let mut peers = self.state.lock().peers.clone();
        peers.shuffle(&mut rand::thread_rng());
        peers
    }
}

/// The URL a snapshot fetch goes to, carrying the caller's freshness
/// requirement when it has one.
fn snapshot_url(peer_base_url: &str, max_age: Option<Duration>) -> String {
    let base = format!("{}{}", peer_base_url.trim_end_matches('/'), SNAPSHOT_PATH);
    match max_age {
        // Saturating rather than wrapping: an absurd duration must read as "any
        // age will do", never as a near-zero age that forces a needless rebuild.
        Some(d) => {
            let ms = u64::try_from(d.as_millis()).unwrap_or(u64::MAX);
            format!("{base}?{MAX_AGE_PARAM}={ms}")
        }
        None => base,
    }
}

/// The URL a cursors-only fetch goes to. Carries no freshness requirement: the
/// producer reads its cursors live on this path, so there is no cached
/// generation to negotiate against. An empty base yields a path-relative URI,
/// usable against an in-process router.
pub(crate) fn cursors_url(peer_base_url: &str) -> String {
    format!(
        "{}{}?{CURSORS_ONLY_PARAM}=true",
        peer_base_url.trim_end_matches('/'),
        SNAPSHOT_PATH,
    )
}

/// What a snapshot fetch got back.
pub enum FetchAnswer {
    Body(PeerSnapshot),
    /// The peer answered non-success. The status distinguishes "an older
    /// router image that does not serve this route" (404) from a sick peer
    /// (5xx) — both retriable, but the first-occurrence log should name which.
    NoBody(reqwest::StatusCode),
}

/// Fetch one peer's snapshot. [`FetchAnswer::NoBody`] means "peer reachable
/// but has no snapshot to give" (non-200, including an older image's 404).
///
/// `max_age` is the oldest export the caller accepts (see
/// [`PRODUCER_CACHE_TTL`]); `None` sends no requirement, as an older consumer
/// does. An older producer ignores the parameter.
pub async fn fetch_snapshot(
    http: &reqwest::Client,
    peer_base_url: &str,
    max_age: Option<Duration>,
) -> Result<FetchAnswer, anyhow::Error> {
    fetch_body(http, &snapshot_url(peer_base_url, max_age), peer_base_url).await
}

/// Fetch one peer's cursor table (`?cursors_only=true`). A producer that
/// ignores the parameter answers with a full snapshot, whose cursor table
/// lists only ranks still carrying nodes; both decode to [`PeerSnapshot`].
/// `None` on non-success.
pub async fn fetch_cursors(
    http: &reqwest::Client,
    peer_base_url: &str,
) -> Result<Option<PeerSnapshot>, anyhow::Error> {
    Ok(
        match fetch_body(http, &cursors_url(peer_base_url), peer_base_url).await? {
            FetchAnswer::Body(snap) => Some(snap),
            FetchAnswer::NoBody(_) => None,
        },
    )
}

/// Shared transport for both fetch shapes: send, branch on what the peer
/// actually encoded, and decode off the runtime.
///
/// `peer_base_url` is carried separately from `url` only so the log names the
/// peer rather than a URL with a query string on it.
///
/// Asks for gzip on this request alone — the snapshot is the largest body the
/// router transfers — rather than via reqwest's `gzip` feature, which is
/// crate-wide and would make every client, including the SSE proxy, advertise
/// and auto-decode it.
async fn fetch_body(
    http: &reqwest::Client,
    url: &str,
    peer_base_url: &str,
) -> Result<FetchAnswer, anyhow::Error> {
    let resp = http
        .get(url)
        .header(reqwest::header::ACCEPT_ENCODING, "gzip")
        .send()
        .await?;
    if !resp.status().is_success() {
        debug!(
            peer = %peer_base_url,
            status = %resp.status(),
            "kv-bootstrap: peer returned no usable body for the snapshot request",
        );
        return Ok(FetchAnswer::NoBody(resp.status()));
    }
    // Branch on what the peer actually sent, not on what we asked for: a router
    // image that predates route compression answers identity, and must keep
    // working. reqwest leaves both the header and the body untouched here because
    // its `gzip` feature is off.
    //
    // Exact-match on purpose. The only legitimate producer is another router of
    // this codebase, which sends the bare token, so a compound or non-canonical
    // coding means an intermediary rewrote the body — better surfaced as a decode
    // failure than guessed at.
    let gzipped = resp
        .headers()
        .get(reqwest::header::CONTENT_ENCODING)
        .is_some_and(|v| v.as_bytes().eq_ignore_ascii_case(b"gzip"));
    // Bounded before buffering; see [`MAX_INFLATED_SNAPSHOT_BYTES`].
    let body = read_bounded(resp, peer_base_url).await?;
    // Inflating and parsing a large tree is seconds of CPU with no await point,
    // on the runtime that also proxies requests (including SSE), so it runs on
    // the blocking pool. A cursors-only body is cheap, but the full-snapshot
    // answer an older producer gives to the same request is not, and one
    // transport path is worth more than saving a task hop on the cheap case.
    tokio::task::spawn_blocking(move || decode_snapshot(&body, gzipped))
        .await?
        .map(FetchAnswer::Body)
}

/// Most a declared `Content-Length` may pre-allocate for the response buffer.
/// The header is peer-controlled, so it sizes the first allocation only up to
/// this; a larger body grows the buffer as it streams in.
const READ_PREALLOC_CAP: u64 = 64 * 1024 * 1024;

/// Most a gzip trailer's ISIZE may pre-allocate for the inflate buffer. The
/// trailer is peer-controlled and records the size modulo 2^32, so it is only
/// a hint; the `take` bound in [`decode_snapshot`] still enforces the ceiling.
const INFLATE_PREALLOC_CAP: u64 = 256 * 1024 * 1024;

/// The error for a body that would pass [`MAX_INFLATED_SNAPSHOT_BYTES`].
fn past_ceiling(what: std::fmt::Arguments<'_>) -> anyhow::Error {
    anyhow::anyhow!(
        "{what} past the {MAX_INFLATED_SNAPSHOT_BYTES}-byte ceiling; refusing to buffer it"
    )
}

/// Buffer a response body, refusing one that grows past
/// [`MAX_INFLATED_SNAPSHOT_BYTES`].
async fn read_bounded(
    mut resp: reqwest::Response,
    peer_base_url: &str,
) -> Result<bytes::Bytes, anyhow::Error> {
    let declared = resp.content_length();
    if declared.is_some_and(|n| n > MAX_INFLATED_SNAPSHOT_BYTES) {
        return Err(past_ceiling(format_args!(
            "peer {peer_base_url} declares a snapshot"
        )));
    }
    let prealloc = declared.map_or(0, |n| n.min(READ_PREALLOC_CAP)) as usize;
    let mut buf = bytes::BytesMut::with_capacity(prealloc);
    // Counted across chunks as well, because `Content-Length` is absent on a
    // chunked response.
    while let Some(chunk) = resp.chunk().await? {
        if buf.len() as u64 + chunk.len() as u64 > MAX_INFLATED_SNAPSHOT_BYTES {
            return Err(past_ceiling(format_args!("peer {peer_base_url} streamed")));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf.freeze())
}

/// Inflate (when gzipped) and parse one snapshot body. Blocking and CPU-bound —
/// callers run it off the async runtime.
fn decode_snapshot(body: &[u8], gzipped: bool) -> Result<PeerSnapshot, anyhow::Error> {
    if !gzipped {
        return Ok(serde_json::from_slice(body)?);
    }
    // The gzip trailer's last four bytes are ISIZE, the inflated length
    // (little-endian, modulo 2^32).
    let isize_hint = body
        .last_chunk::<4>()
        .map_or(0, |b| u64::from(u32::from_le_bytes(*b)));
    let prealloc = isize_hint
        .min(MAX_INFLATED_SNAPSHOT_BYTES)
        .min(INFLATE_PREALLOC_CAP) as usize;
    let mut inflated = Vec::with_capacity(prealloc);
    // Bounded before buffering; see [`MAX_INFLATED_SNAPSHOT_BYTES`].
    let read = GzDecoder::new(body)
        .take(MAX_INFLATED_SNAPSHOT_BYTES + 1)
        .read_to_end(&mut inflated)?;
    if read as u64 > MAX_INFLATED_SNAPSHOT_BYTES {
        return Err(past_ceiling(format_args!("peer snapshot inflates")));
    }
    // Parse from the buffered slice rather than streaming the decoder into
    // serde: serde_json parses a slice much faster than a reader.
    Ok(serde_json::from_slice(&inflated)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::kv_events::Tiers;

    fn wire_worker(url: &str, dp_rank: u32) -> WireWorker {
        WireWorker {
            url: url.into(),
            dp_rank,
        }
    }

    fn node(parent: Option<u32>, block_hash: i64, workers: Vec<u32>) -> SnapshotNode {
        SnapshotNode {
            parent,
            block_hash,
            workers,
            tiers: vec![],
        }
    }

    fn snapshot(workers: Vec<WireWorker>, nodes: Vec<SnapshotNode>) -> PeerSnapshot {
        PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size: 64,
            is_bigram: false,
            producer_ready: true,
            workers,
            cursors: vec![],
            nodes,
        }
    }

    fn live(ids: &[(&str, u32)]) -> HashSet<KvWorkerId> {
        ids.iter()
            .map(|(u, r)| KvWorkerId::new((*u).to_string(), *r))
            .collect()
    }

    /// A caller with no freshness requirement must send a bare path, so an older
    /// router image sees the request it has always seen.
    #[test]
    fn snapshot_url_omits_the_parameter_when_any_age_will_do() {
        assert_eq!(
            snapshot_url("http://peer:3000", None),
            format!("http://peer:3000{SNAPSHOT_PATH}"),
        );
        // Trailing slash on the base must not produce a doubled separator.
        assert_eq!(
            snapshot_url("http://peer:3000/", None),
            format!("http://peer:3000{SNAPSHOT_PATH}"),
        );
    }

    /// A cursors-only fetch carries the parameter that makes the producer skip
    /// its tree.
    #[test]
    fn cursors_url_asks_for_the_cursor_table_alone() {
        assert_eq!(
            cursors_url("http://peer:3000"),
            format!("http://peer:3000{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        );
        // Trailing slash on the base must not produce a doubled separator.
        assert_eq!(
            cursors_url("http://peer:3000/"),
            format!("http://peer:3000{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        );
        // An empty base yields the path-relative URI, for in-process routers.
        assert_eq!(
            cursors_url(""),
            format!("{SNAPSHOT_PATH}?{CURSORS_ONLY_PARAM}=true"),
        );
    }

    #[test]
    fn snapshot_url_states_the_requirement_in_milliseconds() {
        assert_eq!(
            snapshot_url("http://peer:3000", Some(Duration::from_millis(1500))),
            format!("http://peer:3000{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=1500"),
        );
        // Zero is the strictest request, not an absent one.
        assert_eq!(
            snapshot_url("http://peer:3000", Some(Duration::ZERO)),
            format!("http://peer:3000{SNAPSHOT_PATH}?{MAX_AGE_PARAM}=0"),
        );
    }

    /// Saturate rather than wrap. A wrapped duration would read as a near-zero
    /// age and force the peer into a fleet-wide rebuild it was never asked for.
    #[test]
    fn snapshot_url_saturates_an_absurd_age() {
        assert_eq!(
            snapshot_url("http://peer:3000", Some(Duration::MAX)),
            format!(
                "http://peer:3000{SNAPSHOT_PATH}?{MAX_AGE_PARAM}={}",
                u64::MAX
            ),
        );
    }

    /// A wire cursor is read by worker identity, without vetting. A worker the
    /// snapshot does not mention yields `None` rather than a misaddressed
    /// cursor from another rank's table slot.
    #[test]
    fn wire_cursor_is_addressed_by_identity_not_table_position() {
        let snap = PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size: 4,
            is_bigram: false,
            producer_ready: true,
            workers: vec![wire_worker("http://w1", 0), wire_worker("http://w2", 1)],
            // Out of table order, and with no entry for index 0.
            cursors: vec![(1, 77)],
            nodes: vec![],
        };
        assert_eq!(snap.wire_cursor_for("http://w2", 1), Some(77));
        assert_eq!(
            snap.wire_cursor_for("http://w1", 0),
            None,
            "a worker in the table without a cursor must not borrow another's",
        );
        assert_eq!(
            snap.wire_cursor_for("http://w2", 0),
            None,
            "dp_rank matters"
        );
        assert_eq!(snap.wire_cursor_for("http://nope", 0), None);
    }

    #[test]
    fn vet_rejects_unknown_format() {
        let mut snap = snapshot(vec![], vec![]);
        snap.format = SNAPSHOT_FORMAT + 1;
        let err = VettedSnapshot::from_wire(snap, &live(&[]), Some(64)).unwrap_err();
        assert_eq!(
            err,
            VetError::UnknownFormat {
                got: SNAPSHOT_FORMAT + 1,
                want: SNAPSHOT_FORMAT
            }
        );
        assert_eq!(err.outcome(), SnapshotOutcome::Rejected);
    }

    /// A producer that reports not-ready is refused even when its tree holds
    /// nodes.
    #[test]
    fn vet_rejects_a_not_ready_producer_even_with_nodes() {
        let mut snap = snapshot(
            vec![wire_worker("http://w1:30000", 0)],
            vec![node(None, 7, vec![0])],
        );
        snap.producer_ready = false;
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://w1:30000", 0)]), Some(64))
            .unwrap_err();
        assert_eq!(err, VetError::ProducerCold);
        assert_eq!(err.outcome(), SnapshotOutcome::ColdPeer);
    }

    /// An out-of-range `parent` from a peer is refused.
    #[test]
    fn vet_rejects_out_of_range_parent_reference() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0]), node(Some(99), 2, vec![0])],
        );
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64))
            .expect_err("an out-of-range parent must be refused");
        assert_eq!(
            err,
            VetError::InvalidParentReference {
                index: 1,
                parent: 99
            },
        );
        assert_eq!(err.outcome(), SnapshotOutcome::Rejected);
    }

    /// A forward `parent` reference is refused.
    #[test]
    fn vet_rejects_forward_parent_reference_on_the_wire() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(Some(1), 1, vec![0]), node(None, 2, vec![0])],
        );
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64))
            .expect_err("a forward parent must be refused");
        assert_eq!(
            err,
            VetError::InvalidParentReference {
                index: 0,
                parent: 1
            }
        );
    }

    /// A `tiers` list that does not pair with `workers` is refused, not
    /// repaired by filtering.
    #[test]
    fn vet_rejects_tiers_that_do_not_pair_with_workers() {
        let mut bad = node(None, 1, vec![0, 1]);
        bad.tiers = vec![Tiers::HOST.bits()]; // one entry for two carriers
        let snap = snapshot(
            vec![wire_worker("http://a", 0), wire_worker("http://b", 0)],
            vec![node(None, 7, vec![0]), bad],
        );
        let err =
            VettedSnapshot::from_wire(snap, &live(&[("http://a", 0), ("http://b", 0)]), Some(64))
                .expect_err("mispaired tiers must be refused, not repaired");
        assert_eq!(err, VetError::TierTableMismatch { index: 1 });
        assert_eq!(err.outcome(), SnapshotOutcome::Rejected);
    }

    /// A snapshot that pruning empties after the cold gate is refused as
    /// `NothingUsable`.
    #[test]
    fn vet_rejects_snapshot_that_prunes_to_nothing() {
        let snap = snapshot(
            vec![wire_worker("http://rogue", 0)],
            vec![node(None, 1, vec![0]), node(Some(0), 2, vec![0])],
        );
        // The only carrier is a worker this replica has never discovered, so
        // every node becomes carrier-less and is pruned.
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://known", 0)]), Some(64))
            .expect_err("a snapshot that prunes to nothing must be refused");
        assert_eq!(
            err,
            VetError::NothingUsable {
                wire_nodes: 2,
                dropped_workers: 1
            },
        );
        // Retriable: the peer may discover our workers moments later.
        assert_eq!(err.outcome(), SnapshotOutcome::ColdPeer);
    }

    /// An empty peer set is only conclusive if siblings were never seen.
    #[test]
    fn transient_empty_peer_set_is_not_conclusive() {
        let r = PeerRegistry::new();
        assert!(!r.known_to_have_no_peers(), "unsynced is never conclusive");

        r.replace(vec![]);
        assert!(
            r.known_to_have_no_peers(),
            "synced + never any peers ⇒ genuinely alone",
        );

        r.replace(vec!["http://sibling:8090".into()]);
        r.replace(vec![]);
        assert!(
            !r.known_to_have_no_peers(),
            "once siblings have been seen, an empty set must be read as transient",
        );
    }

    /// A snapshot with no cursor for any of the given ranks does not cover
    /// them.
    #[test]
    fn covers_any_is_false_when_the_peer_knows_none_of_our_ranks() {
        let mut snap = snapshot(
            vec![wire_worker("http://known", 0)],
            vec![node(None, 1, vec![0])],
        );
        snap.cursors = vec![(0, 7)];
        let vetted =
            VettedSnapshot::from_wire(snap, &live(&[("http://known", 0)]), Some(64)).unwrap();

        assert!(vetted.covers_any(&[KvWorkerId::new("http://known".into(), 0)]));
        assert!(
            !vetted.covers_any(&[KvWorkerId::new("http://other".into(), 0)]),
            "a snapshot with no cursor for our rank cannot bootstrap it",
        );
    }

    /// Re-registering after a `forget` must mint a NEW incarnation, so a
    /// bootstrap task still in flight for the old one is recognisably stale.
    #[test]
    fn reregistration_after_forget_mints_a_new_epoch() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        let first = t.register(std::slice::from_ref(&a));
        let e1 = first[0].1;

        // Re-registering a still-present rank yields no obligation and keeps
        // its incarnation.
        assert!(
            t.register(std::slice::from_ref(&a)).is_empty(),
            "a registered rank's obligation is already held",
        );
        assert_eq!(
            t.epoch_of(&a),
            Some(e1),
            "a live rank keeps its incarnation"
        );

        t.forget(std::slice::from_ref(&a));
        assert_eq!(t.epoch_of(&a), None);
        let second = t.register(std::slice::from_ref(&a));
        assert_ne!(
            second[0].1, e1,
            "remove + re-add must invalidate the previous incarnation",
        );
    }

    /// A producer that claims ready with an empty tree is refused.
    #[test]
    fn vet_rejects_snapshot_with_no_nodes_even_when_producer_claims_ready() {
        let snap = snapshot(vec![wire_worker("http://a", 0)], vec![]);
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap_err();
        assert_eq!(err, VetError::ProducerCold);
        assert_eq!(err.outcome(), SnapshotOutcome::ColdPeer);
    }

    #[test]
    fn vet_rejects_block_size_mismatch() {
        let snap = snapshot(vec![], vec![node(None, 1, vec![])]);
        let err = VettedSnapshot::from_wire(snap, &live(&[]), Some(32)).unwrap_err();
        assert_eq!(
            err,
            VetError::BlockSizeMismatch {
                peer: 64,
                local: 32
            }
        );
    }

    /// The producer's `is_bigram` stamp does not affect vetting.
    #[test]
    fn vet_ignores_the_producer_bigram_stamp() {
        let mut snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0])],
        );
        for stamp in [false, true] {
            snap.is_bigram = stamp;
            VettedSnapshot::from_wire(snap.clone(), &live(&[("http://a", 0)]), Some(64))
                .unwrap_or_else(|e| panic!("is_bigram={stamp} must not affect vetting: {e}"));
        }
    }

    /// Before any worker establishes a block size there is nothing to
    /// contradict the peer, so the snapshot is accepted.
    #[test]
    fn vet_accepts_when_local_block_size_unset() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0])],
        );
        let vetted = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), None).unwrap();
        assert_eq!(vetted.worker_table.len(), 1);
    }

    /// The core trust-boundary test: a peer naming a worker this replica has
    /// never discovered must not be able to introduce it.
    #[test]
    fn vet_drops_unknown_workers_and_remaps_carriers() {
        let snap = snapshot(
            vec![
                wire_worker("http://known", 0),
                wire_worker("http://rogue", 0),
                wire_worker("http://known", 1),
            ],
            vec![node(None, 100, vec![0, 1, 2]), node(Some(0), 200, vec![1])],
        );
        let vetted = VettedSnapshot::from_wire(
            snap,
            &live(&[("http://known", 0), ("http://known", 1)]),
            Some(64),
        )
        .unwrap();

        assert_eq!(vetted.dropped_workers, 1);
        assert_eq!(
            vetted.worker_table,
            vec![
                KvWorkerId::new("http://known".into(), 0),
                KvWorkerId::new("http://known".into(), 1),
            ],
        );
        // Wire indices 0 and 2 survive as 0 and 1; the rogue index vanishes.
        assert_eq!(vetted.nodes[0].workers, vec![0, 1]);
        // The node whose only carrier was the rogue worker is pruned; see
        // `prune_carrier_less`.
        assert_eq!(
            vetted.nodes.len(),
            1,
            "carrier-less leaf must be pruned, not retained as structure",
        );
    }

    /// Interior structure leading to a surviving carrier must be KEPT — pruning
    /// it would detach the carrier and lose the match entirely.
    #[test]
    fn vet_keeps_carrier_less_interior_nodes_on_a_live_path() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![
                node(None, 1, vec![]),     // carrier-less interior
                node(Some(0), 2, vec![]),  // carrier-less interior
                node(Some(1), 3, vec![0]), // the surviving carrier, at depth 3
            ],
        );
        let vetted = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap();
        assert_eq!(vetted.nodes.len(), 3, "path to a carrier must survive");
        assert_eq!(vetted.nodes[2].workers, vec![0]);
        // Parent links must still be backward references after any remap.
        for (i, rec) in vetted.nodes.iter().enumerate() {
            assert!(rec.parent.is_none_or(|p| (p as usize) < i));
        }
    }

    /// The graft-observability split: carried nodes are matchable, structure
    /// nodes are match paths only. And both accessors describe the SURVIVING
    /// population: a carrier nobody knows is already gone from the count.
    #[test]
    fn carrier_counts_split_carrying_nodes_from_kept_structure() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0), wire_worker("http://drained", 0)],
            vec![
                node(None, 1, vec![]),        // interior on a live path: kept structure
                node(Some(0), 2, vec![0, 1]), // carried (by both)
                node(Some(1), 3, vec![1]),    // carried only by the drained worker
            ],
        );
        let vetted = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap();
        // The drained carrier leaves the worker table; its exclusive node is
        // pruned as a carrier-less leaf, and the shared node keeps the live
        // worker as its only carrier.
        assert_eq!(vetted.worker_count(), 1);
        assert_eq!(vetted.dropped_workers(), 1);
        assert_eq!(vetted.carrier_counts(), (1, 1));
    }

    /// Pruning must remap parent indices, not just drop entries.
    #[test]
    fn prune_remaps_parent_indices() {
        let mut vetted = VettedSnapshot {
            worker_table: vec![KvWorkerId::new("http://a".into(), 0)],
            nodes: vec![
                node(None, 10, vec![]),     // 0: dead leaf, pruned
                node(None, 20, vec![]),     // 1: interior on a live path, kept -> 0
                node(Some(1), 30, vec![0]), // 2: carrier, kept -> 1
                node(Some(1), 40, vec![]),  // 3: dead leaf, pruned
            ],
            cursors: vec![],
            dropped_workers: 0,
        };
        assert_eq!(vetted.prune_carrier_less(), 2);
        assert_eq!(vetted.nodes.len(), 2);
        assert_eq!(vetted.nodes[0].block_hash, 20);
        assert_eq!(vetted.nodes[0].parent, None);
        assert_eq!(vetted.nodes[1].block_hash, 30);
        assert_eq!(
            vetted.nodes[1].parent,
            Some(0),
            "surviving child must point at its parent's NEW index",
        );
    }

    #[test]
    fn vet_drops_cursors_for_unknown_workers() {
        let mut snap = snapshot(
            vec![
                wire_worker("http://known", 0),
                wire_worker("http://rogue", 0),
            ],
            vec![node(None, 1, vec![0])],
        );
        snap.cursors = vec![(0, 42), (1, 99)];
        let vetted =
            VettedSnapshot::from_wire(snap, &live(&[("http://known", 0)]), Some(64)).unwrap();
        assert_eq!(
            vetted.cursors,
            vec![(KvWorkerId::new("http://known".into(), 0), 42)],
        );
    }

    /// Out-of-range carrier indices from a malformed peer are dropped.
    #[test]
    fn vet_ignores_out_of_range_carrier_indices() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0, 7])],
        );
        let vetted = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap();
        assert_eq!(vetted.nodes[0].workers, vec![0]);
    }

    /// A node whose only carrier holds no ranked tier is pruned, so it never
    /// reaches the tree.
    #[test]
    fn vet_prunes_nodes_carried_only_on_unranked_tiers() {
        let unranked: u8 = 1 << 7;
        let mut carried = node(Some(0), 2, vec![0]);
        carried.tiers = vec![Tiers::DEVICE.bits()];
        let mut future_tier_only = node(Some(1), 3, vec![0]);
        future_tier_only.tiers = vec![unranked];
        let mut root = node(None, 1, vec![0]);
        root.tiers = vec![Tiers::DEVICE.bits() | unranked];
        let snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![root, carried, future_tier_only],
        );
        let vetted = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap();
        assert_eq!(
            vetted.node_count(),
            2,
            "a node whose only carrier holds no ranked tier must be pruned",
        );
        let tree = HashTree::new();
        assert_eq!(vetted.graft_into(&tree), Ok(2));
        assert_eq!(tree.node_count(), 2, "no carrier-less node may be grafted");
    }

    /// Every carrier on unranked tiers leaves nothing usable, which is refused
    /// like any other snapshot that prunes to nothing.
    #[test]
    fn vet_refuses_a_snapshot_carried_only_on_unranked_tiers() {
        let mut only = node(None, 1, vec![0]);
        only.tiers = vec![0];
        let snap = snapshot(vec![wire_worker("http://a", 0)], vec![only]);
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap_err();
        assert_eq!(
            err,
            VetError::NothingUsable {
                wire_nodes: 1,
                dropped_workers: 0
            },
        );
    }

    /// Two cursors for one worker make its watermark ambiguous.
    #[test]
    fn vet_rejects_duplicate_cursors_for_one_worker() {
        let mut snap = snapshot(
            vec![wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0])],
        );
        snap.cursors = vec![(0, 5), (0, 100)];
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap_err();
        assert_eq!(err, VetError::DuplicateCursor { worker: 0 });
        assert_eq!(err.outcome(), SnapshotOutcome::Rejected);
    }

    /// A worker table that names one live worker twice is malformed.
    #[test]
    fn vet_rejects_a_repeated_wire_identity() {
        let snap = snapshot(
            vec![wire_worker("http://a", 0), wire_worker("http://a", 0)],
            vec![node(None, 1, vec![0, 1])],
        );
        let err = VettedSnapshot::from_wire(snap, &live(&[("http://a", 0)]), Some(64)).unwrap_err();
        assert_eq!(err, VetError::DuplicateWorker { index: 1 });
        assert_eq!(err.outcome(), SnapshotOutcome::Rejected);
    }

    /// Filtering to a subset of workers drops the others' carriers and
    /// cursors, prunes what that strands, and keeps a shared node with only
    /// the kept carrier.
    #[test]
    fn retain_workers_filters_to_pending_ranks_and_prunes_stranded_nodes() {
        let mut snap = snapshot(
            vec![
                wire_worker("http://pending", 0),
                wire_worker("http://done", 0),
            ],
            vec![
                node(None, 1, vec![0, 1]), // shared
                node(Some(0), 2, vec![1]), // only the filtered-out worker
            ],
        );
        snap.cursors = vec![(0, 3), (1, 4)];
        let ids = [("http://pending", 0), ("http://done", 0)];
        let mut vetted = VettedSnapshot::from_wire(snap, &live(&ids), Some(64)).unwrap();

        vetted.retain_workers(&live(&[("http://pending", 0)]));

        let pending = KvWorkerId::new("http://pending".into(), 0);
        let done = KvWorkerId::new("http://done".into(), 0);
        assert_eq!(
            vetted.node_count(),
            1,
            "the done rank's exclusive node is pruned"
        );
        assert_eq!(vetted.nodes[0].workers, vec![0]);
        assert_eq!(vetted.cursor_for(&pending), Some(3));
        assert_eq!(vetted.cursor_for(&done), None);
    }

    #[test]
    fn tracker_settles_when_all_ranks_terminal() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        let b = KvWorkerId::new("http://a".into(), 1);
        t.register(&[a.clone(), b.clone()]);
        assert!(!t.settled());

        t.set(&a, BootstrapState::Recovered);
        assert!(!t.settled(), "one rank still pending");

        t.set(&b, BootstrapState::Failed);
        assert!(t.settled(), "Failed is terminal — cold is a valid outcome");
    }

    #[test]
    fn fetch_cap_round_trips_through_the_tracker() {
        let t =
            BootstrapTracker::new_with_fetch_cap(Duration::from_secs(120), Duration::from_secs(90));
        assert_eq!(t.timeout(), Duration::from_secs(120));
        assert_eq!(t.fetch_cap(), Duration::from_secs(90));
        assert_eq!(
            BootstrapTracker::new(Duration::from_secs(120)).fetch_cap(),
            DEFAULT_SNAPSHOT_FETCH_TIMEOUT_CAP,
            "a tracker built without a cap takes the default",
        );
    }

    #[test]
    fn tracker_settles_on_deadline_with_pending_ranks() {
        let t = BootstrapTracker::new(Duration::ZERO);
        t.register(&[KvWorkerId::new("http://a".into(), 0)]);
        assert!(t.settled(), "zero timeout settles immediately");
    }

    /// An empty tracker is not settled *yet* — it may still be waiting on
    /// workers to be discovered — but its deadline is already armed, so it
    /// cannot wait forever.
    #[test]
    fn tracker_with_no_ranks_is_not_settled_before_deadline() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        assert!(!t.settled());
        assert!(
            t.time_remaining().is_some(),
            "the deadline must be armed at construction, not at first register",
        );
    }

    /// A tracker that is never registered still settles by its deadline.
    #[test]
    fn tracker_never_registered_still_settles() {
        let t = BootstrapTracker::new(Duration::ZERO);
        assert!(t.settled(), "an unregistered tracker must still settle",);
    }

    /// Registration keeps the deadline armed (the re-arm itself is not timed,
    /// to stay independent of the wall clock).
    #[test]
    fn registration_keeps_the_deadline_armed() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        t.register(&[KvWorkerId::new("http://a".into(), 0)]);
        assert!(t.time_remaining().is_some());
        assert!(!t.settled(), "a pending rank must still gate readiness");
    }

    #[test]
    fn disabled_tracker_is_settled_immediately() {
        assert!(BootstrapTracker::disabled().settled());
    }

    #[test]
    fn tracker_forget_removes_state() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        t.register(std::slice::from_ref(&a));
        assert_eq!(t.state_of(&a), Some(BootstrapState::Pending));
        t.forget(std::slice::from_ref(&a));
        assert_eq!(t.state_of(&a), None);
    }

    #[test]
    fn peer_registry_shuffles_without_losing_entries() {
        let r = PeerRegistry::new();
        assert!(r.is_empty());
        let peers: Vec<String> = (0..16).map(|i| format!("http://r{i}")).collect();
        r.replace(peers.clone());
        assert_eq!(r.len(), 16);

        let mut got = r.candidates();
        assert_eq!(got.len(), 16);
        got.sort();
        let mut want = peers;
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn a_synced_empty_set_with_no_history_means_no_siblings() {
        let reg = PeerRegistry::new();
        reg.replace(vec![]);
        assert!(reg.synced());
        assert!(reg.known_to_have_no_peers());
    }

    /// An expiry observed before the first registration does not latch, so the
    /// registration's re-arm still governs `settled()`.
    #[test]
    fn a_premature_settle_does_not_disarm_the_gate() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        t.expire_deadline_for_test();
        assert!(t.settled(), "the construction deadline has expired");

        let rank = KvWorkerId::new("http://w1".into(), 0);
        t.register(std::slice::from_ref(&rank));
        assert!(
            !t.settled(),
            "the first registration re-arms the budget, so a Pending rank must \
             hold readiness again",
        );

        t.set(&rank, BootstrapState::Recovered);
        assert!(
            t.settled(),
            "and it settles normally once the rank resolves"
        );
    }

    /// Once latched after the first registration, a rank discovered later
    /// cannot pull readiness back to 503 — the scale-up case the latch exists
    /// for.
    #[test]
    fn a_latched_tracker_stays_settled_across_a_later_registration() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        t.register(std::slice::from_ref(&a));
        t.set(&a, BootstrapState::Recovered);
        assert!(t.settled());

        t.register(&[KvWorkerId::new("http://b".into(), 0)]);
        assert!(
            t.settled(),
            "a scale-up must not un-ready a serving replica"
        );
    }

    /// A disabled tracker is settled from the start, and a registration that
    /// reaches it anyway must not open a gate peer bootstrap never configured.
    #[test]
    fn a_disabled_tracker_stays_settled_after_register() {
        let t = BootstrapTracker::disabled();
        t.register(&[KvWorkerId::new("http://a".into(), 0)]);
        assert!(t.settled());
    }

    /// `Instant + Duration` panics on overflow; an absurd timeout must mean
    /// "effectively never", not a crash at construction or first registration.
    #[test]
    fn an_absurd_timeout_does_not_panic() {
        let t = BootstrapTracker::new(Duration::MAX);
        t.register(&[KvWorkerId::new("http://a".into(), 0)]);
        assert!(!t.settled());
        assert!(t
            .time_remaining()
            .is_some_and(|d| d > Duration::from_secs(3600)));
    }

    /// One post-gap retry per incarnation: a second gap on the same one is
    /// final, but a remove + re-add is a fresh publisher and earns its own.
    #[test]
    fn retry_after_gap_is_one_shot_per_incarnation() {
        let t = BootstrapTracker::new(Duration::from_secs(3600));
        let a = KvWorkerId::new("http://a".into(), 0);
        let first = t.register(std::slice::from_ref(&a));

        assert_eq!(t.retry_after_gap(&a), None, "Pending is not a gap state");
        t.set(&a, BootstrapState::Failed);
        assert_eq!(t.retry_after_gap(&a), Some(first[0].clone()));
        assert_eq!(t.state_of(&a), Some(BootstrapState::Pending));

        t.set(&a, BootstrapState::Failed);
        assert_eq!(t.retry_after_gap(&a), None, "already retried once");

        t.forget(std::slice::from_ref(&a));
        assert_eq!(
            t.retry_after_gap(&a),
            None,
            "a forgotten rank never retries"
        );
        let second = t.register(std::slice::from_ref(&a));
        t.set(&a, BootstrapState::Failed);
        assert_eq!(
            t.retry_after_gap(&a),
            Some(second[0].clone()),
            "a new incarnation gets its own retry",
        );
    }

    fn sample_snapshot() -> PeerSnapshot {
        PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size: 64,
            is_bigram: true,
            producer_ready: true,
            workers: vec![
                WireWorker {
                    url: "http://a:30000".into(),
                    dp_rank: 0,
                },
                WireWorker {
                    url: "http://a:30000".into(),
                    dp_rank: 1,
                },
            ],
            cursors: vec![(0, 41), (1, 7)],
            nodes: vec![
                SnapshotNode {
                    parent: None,
                    block_hash: 100,
                    workers: vec![0, 1],
                    tiers: vec![1, 2],
                },
                SnapshotNode {
                    parent: Some(0),
                    block_hash: 200,
                    workers: vec![1],
                    tiers: vec![],
                },
            ],
        }
    }

    /// A body with every field populated survives a JSON round trip.
    #[test]
    fn snapshot_round_trips_through_json() {
        let snap = sample_snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        let back: PeerSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back, snap);
    }

    /// A node with no `tiers` field parses as device-only.
    #[test]
    fn a_node_without_tiers_parses_as_device_only() {
        let json = r#"{
            "format": 1,
            "block_size": 64,
            "is_bigram": false,
            "producer_ready": true,
            "workers": [{"url": "http://a:30000", "dp_rank": 0}],
            "cursors": [[0, 12]],
            "nodes": [{"parent": null, "block_hash": 5, "workers": [0]}]
        }"#;
        let snap: PeerSnapshot = serde_json::from_str(json).unwrap();
        assert_eq!(snap.nodes.len(), 1);
        assert!(snap.nodes[0].tiers.is_empty());
    }

    /// The consumer decodes both encodings a producer can send: identity from
    /// an image that predates route compression, gzip from one that has it.
    #[test]
    fn decode_snapshot_reads_identity_and_gzip_bodies() {
        use std::io::Write;

        let snap = sample_snapshot();
        let json = serde_json::to_vec(&snap).unwrap();
        assert_eq!(decode_snapshot(&json, false).unwrap(), snap);

        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(&json).unwrap();
        let gz = enc.finish().unwrap();
        assert_eq!(decode_snapshot(&gz, true).unwrap(), snap);

        assert!(
            decode_snapshot(&gz, false).is_err(),
            "gzip bytes labelled identity must fail to parse, not half-decode",
        );
        assert!(
            decode_snapshot(&json, true).is_err(),
            "identity bytes labelled gzip must fail to inflate",
        );
    }

    /// A rank the producer does not list, or lists without a cursor, has no
    /// cursor.
    #[test]
    fn wire_cursor_for_resolves_by_wire_identity() {
        let mut snap = sample_snapshot();
        assert_eq!(snap.wire_cursor_for("http://a:30000", 0), Some(41));
        assert_eq!(snap.wire_cursor_for("http://a:30000", 1), Some(7));
        assert_eq!(snap.wire_cursor_for("http://a:30000", 2), None);
        assert_eq!(snap.wire_cursor_for("http://b:30000", 0), None);

        snap.cursors.retain(|(i, _)| *i != 1);
        assert_eq!(
            snap.wire_cursor_for("http://a:30000", 1),
            None,
            "a listed worker with no cursor entry is not a witness",
        );
    }
}
