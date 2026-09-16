// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes EndpointSlice discovery backend.
//!
//! Watches `EndpointSlice` resources by label selector in the configured
//! namespace, diffing against in-memory state and emitting
//! [`DiscoveryEvent`]s for the worker manager.

use crate::config::{K8sDiscoveryConfig, K8sDiscoveryMode};
use crate::discovery::{DiscoveryEvent, WorkerId, WorkerMode, WorkerSpec};
use crate::state::kv_events::bootstrap::PeerRegistry;
use anyhow::{Context, Result};
use futures::{Stream, StreamExt};
use k8s_openapi::api::discovery::v1::{Endpoint, EndpointSlice};
use kube::{
    api::Api,
    runtime::{watcher, WatchStreamExt},
    Client,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Decide which [`WorkerMode`] an `EndpointSlice` should be assigned, based
/// on the configured discovery mode.
///
/// * `Plain` mode — every slice yields `Some(WorkerMode::Plain)`. The
///   server-side label selector has already filtered to the right set.
/// * `PD` mode — the slice's labels are matched against `prefill_selector`
///   and `decode_selector` (in that order). Returns `Some(Prefill)` /
///   `Some(Decode)` on the first match, `None` if neither matches.  The
///   server-side watch in PD mode is unfiltered (label selectors for the
///   two roles may not be mergeable into one Kubernetes selector), so this
///   client-side classification is the gate that drops irrelevant slices.
fn classify_mode(es: &EndpointSlice, mode: &K8sDiscoveryMode) -> Option<WorkerMode> {
    match mode {
        K8sDiscoveryMode::Plain { .. } => Some(WorkerMode::Plain),
        K8sDiscoveryMode::PdDisaggregation {
            prefill_selector,
            decode_selector,
        } => {
            let labels = es.metadata.labels.as_ref().cloned().unwrap_or_default();
            if labels_match_selector(&labels, prefill_selector) {
                Some(WorkerMode::Prefill)
            } else if labels_match_selector(&labels, decode_selector) {
                Some(WorkerMode::Decode)
            } else {
                None
            }
        }
    }
}

/// Match a Kubernetes label set against a comma-separated equality
/// selector like `"app=sglang,role=prefill"`.
///
/// Supports the equality-based subset of the K8s label-selector grammar:
/// `key=value` (and the alias `key==value`). Set-based operators (`in`,
/// `not in`, presence tests) are not supported — fall back to the
/// server-side selector in plain mode if you need them.
fn labels_match_selector(labels: &BTreeMap<String, String>, selector: &str) -> bool {
    for term in selector.split(',') {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }
        // Accept both `=` and `==` for equality.
        let (key, expected) = if let Some((k, v)) = term.split_once("==") {
            (k.trim(), v.trim())
        } else if let Some((k, v)) = term.split_once('=') {
            (k.trim(), v.trim())
        } else {
            // Set-based or presence-only term — not supported here.
            return false;
        };
        match labels.get(key) {
            Some(v) if v == expected => {}
            _ => return false,
        }
    }
    true
}

/// Convert an `EndpointSlice` into a list of [`WorkerSpec`]s with the
/// supplied [`WorkerMode`].
///
/// Skips endpoints whose `conditions.ready` is explicitly `Some(false)`.
/// Per the EndpointSlice API spec, `conditions.ready = None` (absent) means
/// the endpoint should be considered ready.
///
/// The worker URL is `http://<addr>:<port>` where port comes from
/// `EndpointSlice.ports[0].port`, defaulting to `30000` if absent.
///
/// The [`WorkerId`] is `{ns}/{uid}` where `uid` is the pod's K8s UID
/// from `endpoint.target_ref.uid`. Pod UIDs are globally unique per
/// pod incarnation, so when a pod dies and a new pod gets the same IP
/// (kubelet IP reuse on a busy podCIDR), the router sees a fresh
/// `WorkerId` and emits a clean Removed→Added cycle — old breaker /
/// active-load state is shed instead of being mis-applied to the new
/// pod. For manually-created EndpointSlices that lack `target_ref`
/// (rare in real clusters but common in unit tests), falls back to
/// `{ns}/{slice_name}/{addr}:{port}`.
///
/// `model_ids` is intentionally left empty — model membership is resolved
/// by the worker manager via `/server_info` introspection after the
/// `Added` event is emitted.
fn extract_workers(es: &EndpointSlice, mode: WorkerMode) -> Vec<WorkerSpec> {
    let port = es
        .ports
        .as_ref()
        .and_then(|p| p.first())
        .and_then(|p| p.port)
        .unwrap_or(30000);

    let ns = es.metadata.namespace.as_deref().unwrap_or("");
    let slice_name = es.metadata.name.as_deref().unwrap_or("");

    let mut out = Vec::new();
    for ep in es.endpoints.iter() {
        if !endpoint_ready(ep) {
            continue;
        }
        let pod_uid: Option<&str> = ep.target_ref.as_ref().and_then(|r| r.uid.as_deref());
        for addr in &ep.addresses {
            let host = if es.address_type == "IPv6" {
                format!("[{addr}]")
            } else {
                addr.clone()
            };
            let url = format!("http://{host}:{port}");
            let id = match pod_uid {
                Some(uid) => WorkerId(format!("{ns}/{uid}")),
                None => WorkerId(format!("{ns}/{slice_name}/{addr}:{port}")),
            };
            // bootstrap_port stays `None` here on purpose. The final
            // `WorkerMode` and the bootstrap port are both filled in by
            // the worker manager from each worker's `/server_info`
            // body (`disaggregation_mode` + `disaggregation_bootstrap_port`,
            // both fields on SGLang's ServerArgs and already surfaced
            // via `**asdict(server_args)` in the response). EndpointSlice
            // carries neither, but doesn't need to — see
            // `src/workers/introspect.rs` for the extraction and
            // `register_one` in `src/workers/manager.rs` for the override.
            out.push(WorkerSpec {
                id,
                url,
                mode,
                model_ids: Vec::new(),
                bootstrap_port: None,
            });
        }
    }
    out
}

/// Per the EndpointSlice API, an absent `conditions.ready` means ready.
fn endpoint_ready(ep: &Endpoint) -> bool {
    ep.conditions.as_ref().and_then(|c| c.ready).unwrap_or(true)
}

/// Spawn the k8s discovery task.
///
/// Connects to the cluster via `KUBECONFIG` / in-cluster service account,
/// then watches `EndpointSlice` resources in `cfg.namespace`. Diffs against
/// in-memory state and emits [`DiscoveryEvent`]s to `tx`.
///
/// In plain mode the configured `label_selector` is pushed to the server
/// side; in PD mode the watch is unfiltered and per-slice classification
/// happens client-side via `classify_mode`.
///
/// Stable per-slice key.
///
/// Uses `metadata.uid` when present (the normal case in a real cluster), and
/// falls back to `{ns}/{name}` for slices without a UID (rare: CR shims,
/// tests, certain fake/in-memory backends).  The fallback is unique per slice
/// because EndpointSlice names are unique within a namespace.
fn slice_key(es: &EndpointSlice) -> String {
    if let Some(uid) = es.metadata.uid.as_deref() {
        if !uid.is_empty() {
            return uid.to_string();
        }
    }
    let ns = es.metadata.namespace.as_deref().unwrap_or("");
    let name = es.metadata.name.as_deref().unwrap_or("");
    format!("{ns}/{name}")
}

/// Send all `Added` / `Removed` / `ModeChanged` events that bring the
/// consumer from `prev_union` to the recomputed union of `per_slice`.
///
/// Returns `Err` on the first send failure (consumer dropped); the caller is
/// expected to exit the watcher loop.  Updates `prev_union` in place to the
/// new union on success.
async fn emit_diff(
    tx: &mpsc::Sender<DiscoveryEvent>,
    per_slice: &HashMap<String, HashMap<WorkerId, WorkerSpec>>,
    prev_union: &mut HashMap<WorkerId, WorkerSpec>,
) -> Result<(), mpsc::error::SendError<DiscoveryEvent>> {
    let union: HashMap<WorkerId, WorkerSpec> = per_slice
        .values()
        .flat_map(|s| s.iter().map(|(k, v)| (k.clone(), v.clone())))
        .collect();

    for (id, spec) in &union {
        match prev_union.get(id) {
            Some(prev) => {
                if prev.mode != spec.mode {
                    tx.send(DiscoveryEvent::ModeChanged {
                        id: id.clone(),
                        mode: spec.mode,
                    })
                    .await?;
                }
                if prev.url != spec.url || prev.model_ids != spec.model_ids {
                    tx.send(DiscoveryEvent::Removed { id: id.clone() }).await?;
                    tx.send(DiscoveryEvent::Added(spec.clone())).await?;
                }
            }
            None => {
                tx.send(DiscoveryEvent::Added(spec.clone())).await?;
            }
        }
    }

    let dropped: Vec<WorkerId> = prev_union
        .keys()
        .filter(|id| !union.contains_key(id))
        .cloned()
        .collect();
    for id in dropped {
        tx.send(DiscoveryEvent::Removed { id }).await?;
    }

    *prev_union = union;
    Ok(())
}

/// Drive the event-processing loop for a stream of `watcher::Event`s.
///
/// Handles the full set of `kube` watcher events:
/// * `Init` / `InitApply` / `InitDone` — full-LIST resync.  Objects are
///   buffered until `InitDone`, then swapped in atomically.  Any slices that
///   were present before but not seen during the resync are diffed out as
///   `Removed`, which catches deletions that occurred while the watcher was
///   disconnected.
/// * `Apply` — single-object upsert into `per_slice`.
/// * `Delete` — single-object removal from `per_slice`; the diff emits
///   `Removed` for every worker that lived in that slice.
///
/// On any watcher error the kube-runtime watcher auto-restarts and emits a
/// new `Init` cycle, so the resync logic above is what reconciles state
/// after transient errors — no separate state reset is required.
///
/// The loop returns when the input stream ends (logged at WARN) or when the
/// consumer drops the receiving end of `tx` (logged at INFO).
async fn process_events<S>(mut stream: S, tx: mpsc::Sender<DiscoveryEvent>, mode: K8sDiscoveryMode)
where
    S: Stream<Item = Result<watcher::Event<EndpointSlice>, watcher::Error>> + Unpin,
{
    let mut per_slice: HashMap<String, HashMap<WorkerId, WorkerSpec>> = HashMap::new();
    let mut prev_union: HashMap<WorkerId, WorkerSpec> = HashMap::new();
    let mut init_buffer: Option<HashMap<String, HashMap<WorkerId, WorkerSpec>>> = None;

    fn workers_for_slice(
        es: &EndpointSlice,
        mode: &K8sDiscoveryMode,
    ) -> HashMap<WorkerId, WorkerSpec> {
        match classify_mode(es, mode) {
            Some(wm) => extract_workers(es, wm)
                .into_iter()
                .map(|w| (w.id.clone(), w))
                .collect(),
            None => HashMap::new(),
        }
    }

    while let Some(event) = stream.next().await {
        let result = match event {
            Ok(watcher::Event::Init) => {
                init_buffer = Some(HashMap::new());
                Ok(())
            }
            Ok(watcher::Event::InitApply(es)) => {
                let key = slice_key(&es);
                let workers = workers_for_slice(&es, &mode);
                if let Some(buf) = init_buffer.as_mut() {
                    buf.insert(key, workers);
                    Ok(())
                } else {
                    // Defensive: InitApply outside an Init cycle.  Treat as Apply.
                    per_slice.insert(key, workers);
                    emit_diff(&tx, &per_slice, &mut prev_union).await
                }
            }
            Ok(watcher::Event::InitDone) => {
                if let Some(buf) = init_buffer.take() {
                    per_slice = buf;
                    emit_diff(&tx, &per_slice, &mut prev_union).await
                } else {
                    Ok(())
                }
            }
            Ok(watcher::Event::Apply(es)) => {
                let key = slice_key(&es);
                let workers = workers_for_slice(&es, &mode);
                per_slice.insert(key, workers);
                emit_diff(&tx, &per_slice, &mut prev_union).await
            }
            Ok(watcher::Event::Delete(es)) => {
                let key = slice_key(&es);
                per_slice.remove(&key);
                emit_diff(&tx, &per_slice, &mut prev_union).await
            }
            Err(e) => {
                tracing::warn!(error = ?e, "k8s watcher error; awaiting auto-restart");
                Ok(())
            }
        };
        if result.is_err() {
            tracing::info!("k8s discovery: event channel closed; exiting watcher");
            return;
        }
    }
    tracing::warn!("k8s watcher stream ended; discovery task exiting");
}

/// An `EndpointSlice` API handle from the default client config.
///
/// An empty `namespace` means a cluster-wide watch via `Api::all(client)`.
/// `Api::namespaced(client, "")` is namespace-scoped to the empty-named
/// namespace, which is almost never what callers intend.
async fn endpoint_slice_api(namespace: &str) -> Result<Api<EndpointSlice>, kube::Error> {
    let client = Client::try_default().await?;
    Ok(if namespace.is_empty() {
        Api::all(client)
    } else {
        Api::namespaced(client, namespace)
    })
}

/// How a watched namespace appears in startup logs.
fn namespace_display(namespace: &str) -> &str {
    if namespace.is_empty() {
        "<all namespaces>"
    } else {
        namespace
    }
}

/// State is tracked per-slice as `HashMap<SliceKey, HashMap<WorkerId,
/// WorkerSpec>>`.  K8s auto-shards Services with >100 endpoints and CNIs
/// often shard per AZ, so multiple `EndpointSlice` objects can exist per
/// Service; a flat state map would let each slice's event silently drop all
/// workers from sibling slices.  The global union is recomputed from all
/// submaps on every event and diffed against `prev_union` to produce
/// `DiscoveryEvent`s.
///
/// The returned `JoinHandle` runs until the channel is closed or the watcher
/// stream ends (server restart, RBAC change, etc.).
pub async fn spawn(
    cfg: K8sDiscoveryConfig,
    tx: mpsc::Sender<DiscoveryEvent>,
) -> Result<tokio::task::JoinHandle<()>> {
    // The mode was resolved + validated at construction (`resolve_mode` in
    // `Cli::build_discovery`); just destructure it here.
    // `peer_selector` names sibling routers, not the workers streamed here.
    let K8sDiscoveryConfig {
        namespace,
        mode,
        peer_selector: _,
    } = cfg;

    let api = endpoint_slice_api(&namespace)
        .await
        .context("kube client default config")?;

    // Plain mode pushes the single selector to the server side so the LIST
    // is already filtered.  PD mode leaves the server-side selector empty
    // because the prefill/decode selectors may not be expressible as one
    // K8s label-selector — classification happens client-side per slice
    // via `classify_mode`.
    let server_side_selector = match &mode {
        K8sDiscoveryMode::Plain { label_selector } => label_selector.clone(),
        K8sDiscoveryMode::PdDisaggregation { .. } => String::new(),
    };
    let watcher_cfg = watcher::Config::default().labels(&server_side_selector);

    // Log the resolved namespace + selector(s) at startup. We can't
    // verify the namespace exists (the router's RBAC covers
    // endpointslices/services/pods, not namespaces, and a correct
    // namespace legitimately has zero matching workers until they come
    // up), so a typo'd `--service-discovery-namespace` silently watches
    // an empty namespace. Surfacing the watch target here lets an
    // operator spot the typo in the first log lines instead of only
    // discovering it via later `no workers available` request failures.
    let namespace_display = namespace_display(&namespace);
    match &mode {
        K8sDiscoveryMode::Plain { label_selector } => tracing::info!(
            namespace = %namespace_display,
            label_selector = %label_selector,
            "k8s discovery starting (plain mode); a wrong namespace or selector matches zero EndpointSlices"
        ),
        K8sDiscoveryMode::PdDisaggregation {
            prefill_selector,
            decode_selector,
        } => tracing::info!(
            namespace = %namespace_display,
            prefill_selector = %prefill_selector,
            decode_selector = %decode_selector,
            "k8s discovery starting (PD mode); a wrong namespace or selector matches zero EndpointSlices"
        ),
    }

    let handle = tokio::spawn(async move {
        let stream = watcher(api, watcher_cfg);
        tokio::pin!(stream);
        process_events(stream, tx, mode).await;
    });
    Ok(handle)
}

// ---------------------------------------------------------------------------
// Peer discovery — sibling router replicas
// ---------------------------------------------------------------------------

/// The address family sibling URLs must use: the one this router listens on.
///
/// A dual-stack Service has one slice per family, and siblings listen as this
/// replica does, so only the family its listener accepts is reachable.
/// `0.0.0.0` opens an `AF_INET` socket, so it is IPv4-only however
/// "unspecified" it looks. Only `::` is a dual-stack listener (Linux leaves
/// `IPV6_V6ONLY` off by default), and a hostname names no family; both are
/// `Any`. Reading `::` as IPv6-only would discard every slice on a
/// single-stack IPv4 cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddressFamily {
    V4,
    V6,
    Any,
}

impl AddressFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V4 => "IPv4",
            Self::V6 => "IPv6",
            Self::Any => "any",
        }
    }
}

/// The [`AddressFamily`] of a router bound to `host`.
pub fn peer_address_family(host: &str) -> AddressFamily {
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(_)) => AddressFamily::V4,
        Ok(std::net::IpAddr::V6(ip)) if !ip.is_unspecified() => AddressFamily::V6,
        Ok(std::net::IpAddr::V6(_)) | Err(_) => AddressFamily::Any,
    }
}

/// How this replica recognizes itself in an EndpointSlice, so its own pod is
/// kept out of the peer set.
///
/// Built from the downward API. `pod_name` is the primary signal: it matches
/// the endpoint's `target_ref` and so excludes ALL of the pod's addresses,
/// which `ip` alone cannot do — `POD_IP` carries only the pod's primary
/// address, so on a dual-stack Service the pod's secondary-family address
/// would survive an IP-only filter. `ip` is kept as a fallback for endpoints
/// whose `target_ref` is absent (manually-created slices).
///
/// Without `POD_NAME`, `pod_name` falls back to `HOSTNAME`, as the access
/// log's pod identity does: Kubernetes defaults a pod's hostname to its name.
/// A hostname that is not the pod name (`hostNetwork`, a set `spec.hostname`)
/// just matches nothing.
#[derive(Default, Debug)]
struct SelfIdentity {
    pod_name: Option<String>,
    pod_namespace: Option<String>,
    ip: Option<String>,
}

/// A set-but-empty env var must count as unset: the downward API never yields
/// empty, but a hand-written manifest can, and `Some("")` would suppress the
/// [`SelfIdentity::is_unknown`] warning while matching nothing.
fn env_non_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|s| !s.is_empty())
}

impl SelfIdentity {
    fn from_env() -> Self {
        Self {
            pod_name: env_non_empty("POD_NAME").or_else(|| env_non_empty("HOSTNAME")),
            pod_namespace: env_non_empty("POD_NAMESPACE"),
            ip: env_non_empty("POD_IP"),
        }
    }

    /// True when nothing usable for self-exclusion is known (`POD_NAMESPACE`
    /// alone cannot exclude anything). This replica then stays in its own
    /// peer set.
    fn is_unknown(&self) -> bool {
        self.pod_name.is_none() && self.ip.is_none()
    }

    /// Does this endpoint belong to this replica's own pod?
    ///
    /// An endpoint is one backend and its addresses are fungible (the
    /// EndpointSlice API says so), so one address matching `POD_IP` makes the
    /// whole endpoint this replica.
    fn matches_endpoint(&self, ep: &Endpoint) -> bool {
        let self_ip = self.ip.as_deref();
        if ep.addresses.iter().any(|a| Some(a.as_str()) == self_ip) {
            return true;
        }
        let (Some(name), Some(tref)) = (self.pod_name.as_deref(), ep.target_ref.as_ref()) else {
            return false;
        };
        // Pod-backed endpoints always carry kind/name; a missing kind on a
        // manually-written slice is treated as Pod rather than never matching.
        if tref.kind.as_deref().unwrap_or("Pod") != "Pod" || tref.name.as_deref() != Some(name) {
            return false;
        }
        // When our own namespace is known, a same-named pod elsewhere is not
        // us. When it is unknown, a bare name match can only false-positive on
        // an identical pod name in another watched namespace — contrived for
        // most workloads, though deterministic for same-named StatefulSets,
        // and impossible once POD_NAMESPACE is set. A target_ref with no
        // namespace is treated as same-namespace: excluding beats listing this
        // replica as its own peer.
        match (self.pod_namespace.as_deref(), tref.namespace.as_deref()) {
            (Some(want), Some(got)) => want == got,
            _ => true,
        }
    }
}

/// One sibling as one slice lists it.
#[derive(Debug)]
struct PeerAddr {
    /// The backend behind `url`, so a sibling listed once per address family
    /// (a dual-stack Service has one slice per family) is offered once: the
    /// target pod's UID and the port, else the URL itself.
    backend: String,
    url: String,
}

/// What this replica keeps from each EndpointSlice of its own Service.
#[derive(Debug)]
struct PeerFilter {
    identity: SelfIdentity,
    family: AddressFamily,
    /// This replica's listen port.
    port: i32,
}

impl PeerFilter {
    /// Extract sibling replicas from an EndpointSlice.
    ///
    /// Only `ready` endpoints are considered: an unready sibling is either
    /// still starting (its tree is cold) or draining. This replica's own
    /// endpoint is excluded (see [`SelfIdentity`]), and so is a slice of the
    /// other [`AddressFamily`]. Each endpoint yields its FIRST address only:
    /// the API defines an endpoint's addresses as fungible, so the rest name
    /// the same backend.
    fn extract(&self, es: &EndpointSlice) -> Vec<PeerAddr> {
        // FQDN and unrecognised address types are kept.
        let family_matches = !matches!(
            (self.family, es.address_type.as_str()),
            (AddressFamily::V4, "IPv6") | (AddressFamily::V6, "IPv4")
        );
        if !family_matches {
            return Vec::new();
        }
        // Siblings are this router's own pods, so they listen where this
        // replica does: prefer an advertised port equal to our own, and use
        // our own when the slice advertises none. A `metrics` or `grpc` port
        // declared ahead of `http` must not win just by sorting first.
        let ports = es.ports.as_deref().unwrap_or_default();
        let port = ports
            .iter()
            .filter_map(|p| p.port)
            .find(|&p| p == self.port)
            .or_else(|| ports.iter().find_map(|p| p.port))
            .unwrap_or(self.port);
        es.endpoints
            .iter()
            .filter(|ep| endpoint_ready(ep))
            .filter(|ep| {
                // Otherwise silent: this line is the only trace of a sibling
                // misclassified as self.
                if self.identity.matches_endpoint(ep) {
                    tracing::debug!(
                        addrs = ?ep.addresses,
                        target_ref = ?ep.target_ref,
                        "kv-bootstrap: excluding endpoint as self (pod identity match)",
                    );
                    return false;
                }
                true
            })
            .filter_map(|ep| {
                let addr = ep.addresses.first()?;
                // IPv6 literals need brackets in a URL authority.
                let url = if addr.contains(':') {
                    format!("http://[{addr}]:{port}")
                } else {
                    format!("http://{addr}:{port}")
                };
                let backend = match ep.target_ref.as_ref().and_then(|r| r.uid.as_deref()) {
                    Some(uid) if !uid.is_empty() => format!("{uid}:{port}"),
                    _ => url.clone(),
                };
                Some(PeerAddr { backend, url })
            })
            .collect()
    }
}

/// Flatten the per-slice view into the published peer list: one URL per
/// backend, sorted.
///
/// Within a backend the lexicographically first URL wins, which keeps the
/// choice deterministic and prefers IPv4 (its digits sort before IPv6's `[`).
fn peer_urls(by_slice: &HashMap<String, Vec<PeerAddr>>) -> Vec<String> {
    let mut all: Vec<&PeerAddr> = by_slice.values().flatten().collect();
    all.sort_unstable_by(|a, b| (&a.backend, &a.url).cmp(&(&b.backend, &b.url)));
    all.dedup_by(|a, b| a.backend == b.backend);
    let mut urls: Vec<String> = all.into_iter().map(|p| p.url.clone()).collect();
    urls.sort_unstable();
    urls.dedup();
    urls
}

/// Watch this router's own EndpointSlices and keep `peers` current.
///
/// Separate from [`spawn`] because it tracks sibling router replicas rather
/// than engines, and must keep working when the worker watch is in PD mode
/// with client-side classification.
///
/// The task exits when the watcher stream ends; the peer set then stays at its
/// last value. Non-fatal: routing does not depend on this watch.
pub async fn spawn_peer_watch(
    namespace: String,
    label_selector: String,
    peers: Arc<PeerRegistry>,
    family: AddressFamily,
    self_port: i32,
) -> Result<tokio::task::JoinHandle<()>> {
    let api = endpoint_slice_api(&namespace)
        .await
        .context("kube client default config for peer watch")?;
    let filter = PeerFilter {
        identity: SelfIdentity::from_env(),
        family,
        port: self_port,
    };
    if filter.identity.is_unknown() {
        tracing::warn!(
            "kv-bootstrap: none of POD_NAME, HOSTNAME or POD_IP is set, so this replica \
             cannot exclude itself from its peer list; set POD_NAME from the downward API \
             for a cleaner peer set"
        );
    }

    tracing::info!(
        namespace = %namespace_display(&namespace),
        label_selector = %label_selector,
        self_pod_name = %filter.identity.pod_name.as_deref().unwrap_or("<unset>"),
        self_ip = %filter.identity.ip.as_deref().unwrap_or("<unset>"),
        self_port,
        address_family = %family.as_str(),
        "kv-bootstrap: peer watch starting",
    );

    let watcher_cfg = watcher::Config::default().labels(&label_selector);
    let handle = tokio::spawn(async move {
        // The watcher retries on the very next poll after an error, so without
        // a backoff a persistent failure — a 403 from missing RBAC, a 400 from
        // a malformed selector — becomes a tight LIST loop against the API
        // server, one WARN line per iteration, for the life of the process.
        let stream = watcher(api, watcher_cfg).default_backoff();
        tokio::pin!(stream);
        process_peer_events(stream, &peers, &filter).await;
    });
    Ok(handle)
}

/// Drive the peer-set event loop for a stream of `watcher::Event`s.
///
/// Split out from [`spawn_peer_watch`] so the relist bookkeeping is testable
/// over a plain stream, without a cluster.
async fn process_peer_events<S>(mut stream: S, peers: &PeerRegistry, filter: &PeerFilter)
where
    S: Stream<Item = Result<watcher::Event<EndpointSlice>, watcher::Error>> + Unpin,
{
    // A Service's endpoints are commonly sharded across several
    // EndpointSlices (one per AZ, one per address family), which is why this
    // is per-slice rather than a flat list.
    let mut by_slice: HashMap<String, Vec<PeerAddr>> = HashMap::new();
    // Relist buffer, `Some` only between `Init` and `InitDone`, so a relist
    // publishes its complete result at once. Publishing each `InitApply` would
    // expose a partial set, and an empty partial marks the registry synced,
    // making `known_to_have_no_peers()` briefly true. A stray `InitDone`
    // publishes nothing.
    let mut init_buffer: Option<HashMap<String, Vec<PeerAddr>>> = None;
    let publish = |by_slice: &HashMap<String, Vec<PeerAddr>>| peers.replace(peer_urls(by_slice));

    while let Some(event) = stream.next().await {
        match event {
            Ok(watcher::Event::Init) => init_buffer = Some(HashMap::new()),
            Ok(watcher::Event::InitApply(es)) => {
                let key = slice_key(&es);
                let found = filter.extract(&es);
                if let Some(buf) = init_buffer.as_mut() {
                    buf.insert(key, found);
                } else {
                    // Defensive: InitApply outside an Init cycle. Treat as Apply.
                    by_slice.insert(key, found);
                    publish(&by_slice);
                }
            }
            Ok(watcher::Event::InitDone) => {
                let Some(buf) = init_buffer.take() else {
                    continue;
                };
                // Swap atomically: the relist result fully replaces the old
                // view, so slices deleted while the watch was down disappear
                // here.
                by_slice = buf;
                if by_slice.is_empty() {
                    // A correct selector matches at least this replica's own
                    // Service's slice, so zero slices is a misconfiguration,
                    // not a fleet of one — yet it publishes as "alone".
                    tracing::warn!(
                        "kv-bootstrap: the peer selector matched no EndpointSlices; it is \
                         matched against EndpointSlice labels, which are copied from the \
                         router's Service (not its pods), in the watched namespace"
                    );
                }
                publish(&by_slice);
            }
            Ok(watcher::Event::Apply(es)) => {
                by_slice.insert(slice_key(&es), filter.extract(&es));
                publish(&by_slice);
            }
            Ok(watcher::Event::Delete(es)) => {
                by_slice.remove(&slice_key(&es));
                publish(&by_slice);
            }
            Err(e) => {
                tracing::warn!(
                    error = ?e,
                    "kv-bootstrap: peer watcher error; retrying with backoff (a persistent \
                     403 means the ServiceAccount lacks list/watch on endpointslices for \
                     the router's own Service)",
                );
            }
        }
    }
    tracing::warn!("kv-bootstrap: peer watcher stream ended; peer set is now frozen");
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::ObjectReference;
    use k8s_openapi::api::discovery::v1::{EndpointConditions, EndpointPort};
    use kube::core::ObjectMeta;

    /// Helper: build a minimal EndpointSlice with predictable metadata.
    fn make_slice(addrs: &[&str], port: i32, ready: bool) -> EndpointSlice {
        make_slice_full(addrs, port, ready, "testns", "test-slice", &[])
    }

    fn make_slice_ns(
        addrs: &[&str],
        port: i32,
        ready: bool,
        ns: &str,
        slice_name: &str,
    ) -> EndpointSlice {
        make_slice_full(addrs, port, ready, ns, slice_name, &[])
    }

    fn make_slice_with_labels(
        addrs: &[&str],
        port: i32,
        ready: bool,
        labels: &[(&str, &str)],
    ) -> EndpointSlice {
        make_slice_full(addrs, port, ready, "testns", "test-slice", labels)
    }

    fn make_slice_full(
        addrs: &[&str],
        port: i32,
        ready: bool,
        ns: &str,
        slice_name: &str,
        labels: &[(&str, &str)],
    ) -> EndpointSlice {
        let label_map: BTreeMap<String, String> = labels
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        EndpointSlice {
            metadata: ObjectMeta {
                labels: if label_map.is_empty() {
                    None
                } else {
                    Some(label_map)
                },
                name: if slice_name.is_empty() {
                    None
                } else {
                    Some(slice_name.into())
                },
                namespace: if ns.is_empty() { None } else { Some(ns.into()) },
                ..Default::default()
            },
            address_type: "IPv4".into(),
            endpoints: vec![Endpoint {
                addresses: addrs.iter().map(|a| (*a).to_string()).collect(),
                conditions: Some(EndpointConditions {
                    ready: Some(ready),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ports: Some(vec![EndpointPort {
                port: Some(port),
                ..Default::default()
            }]),
        }
    }

    fn plain_mode() -> K8sDiscoveryMode {
        K8sDiscoveryMode::Plain {
            label_selector: "app=sglang".into(),
        }
    }

    fn pd_mode() -> K8sDiscoveryMode {
        K8sDiscoveryMode::PdDisaggregation {
            prefill_selector: "app=sglang,role=prefill".into(),
            decode_selector: "app=sglang,role=decode".into(),
        }
    }

    #[test]
    fn classify_mode_plain_returns_some_plain_for_any_slice() {
        let s = make_slice(&["10.0.0.1"], 30000, true);
        assert_eq!(classify_mode(&s, &plain_mode()), Some(WorkerMode::Plain));
    }

    #[test]
    fn classify_mode_pd_returns_prefill_when_labels_match_prefill_selector() {
        let s = make_slice_with_labels(
            &["10.0.0.1"],
            30000,
            true,
            &[("app", "sglang"), ("role", "prefill")],
        );
        assert_eq!(classify_mode(&s, &pd_mode()), Some(WorkerMode::Prefill));
    }

    #[test]
    fn classify_mode_pd_returns_decode_when_labels_match_decode_selector() {
        let s = make_slice_with_labels(
            &["10.0.0.1"],
            30000,
            true,
            &[("app", "sglang"), ("role", "decode")],
        );
        assert_eq!(classify_mode(&s, &pd_mode()), Some(WorkerMode::Decode));
    }

    #[test]
    fn classify_mode_pd_returns_none_when_no_selector_matches() {
        let s = make_slice_with_labels(
            &["10.0.0.1"],
            30000,
            true,
            &[("app", "sglang"), ("role", "router")],
        );
        assert!(classify_mode(&s, &pd_mode()).is_none());

        // No labels at all => still None in PD mode (every selector term
        // requires a key/value).
        let s = make_slice(&["10.0.0.1"], 30000, true);
        assert!(classify_mode(&s, &pd_mode()).is_none());
    }

    #[test]
    fn extract_workers_emits_workers_with_supplied_mode_and_empty_model_ids() {
        let s = make_slice(&["10.0.0.1"], 30000, true);
        let ws = extract_workers(&s, WorkerMode::Plain);
        assert_eq!(ws.len(), 1);
        assert_eq!(ws[0].mode, WorkerMode::Plain);
        assert_eq!(ws[0].url, "http://10.0.0.1:30000");
        assert_eq!(ws[0].id.0, "testns/test-slice/10.0.0.1:30000");
        assert!(
            ws[0].model_ids.is_empty(),
            "model_ids are resolved via /server_info, not at extract time"
        );

        // The mode argument flows through unchanged.
        let ws = extract_workers(&s, WorkerMode::Prefill);
        assert_eq!(ws[0].mode, WorkerMode::Prefill);
        let ws = extract_workers(&s, WorkerMode::Decode);
        assert_eq!(ws[0].mode, WorkerMode::Decode);
    }

    #[test]
    fn brackets_ipv6_worker_addresses() {
        let mut slice = make_slice(&["2001:db8::1"], 30000, true);
        slice.address_type = "IPv6".into();
        let workers = extract_workers(&slice, WorkerMode::Plain);
        assert_eq!(workers[0].url, "http://[2001:db8::1]:30000");
        assert!(url::Url::parse(&workers[0].url).is_ok());
    }

    #[test]
    fn skips_not_ready_endpoints() {
        let s = make_slice(&["10.0.0.1"], 30000, false);
        assert!(extract_workers(&s, WorkerMode::Plain).is_empty());
    }

    /// `conditions.ready = None` must default to ready=true per EndpointSlice
    /// API spec ("undefined → endpoint should be considered ready").
    #[test]
    fn is_ready_none_defaults_to_ready() {
        let mut s = make_slice(&["10.0.0.1"], 30000, false /* overridden below */);
        s.endpoints[0].conditions.as_mut().unwrap().ready = None;
        let ws = extract_workers(&s, WorkerMode::Plain);
        assert_eq!(
            ws.len(),
            1,
            "endpoint with None ready should be treated as ready"
        );
    }

    /// WorkerId must include namespace + slice name to avoid cross-namespace
    /// IP collisions on overlay networks.
    #[test]
    fn worker_id_includes_namespace_and_slice_name() {
        let mut s = make_slice(&["10.0.0.1"], 30000, true);
        s.metadata.namespace = Some("prod".to_string());
        s.metadata.name = Some("svc-abc-xyz".to_string());
        let ws = extract_workers(&s, WorkerMode::Plain);
        assert_eq!(ws[0].id.0, "prod/svc-abc-xyz/10.0.0.1:30000");
    }

    /// A cluster-scoped slice (no namespace metadata) must still produce a
    /// valid `WorkerId` — empty-namespace prefix yields a leading slash.
    #[test]
    fn worker_id_handles_missing_namespace() {
        // make_slice_ns with empty ns leaves metadata.namespace = None.
        let s = make_slice_ns(&["10.0.0.1"], 30000, true, "", "my-slice");
        let ws = extract_workers(&s, WorkerMode::Plain);
        assert!(
            ws[0].id.0.contains("10.0.0.1:30000"),
            "id must contain addr:port"
        );
        assert!(
            ws[0].id.0.starts_with('/'),
            "empty ns => id starts with '/', got: {}",
            ws[0].id.0
        );
    }

    /// Two slices with no `metadata.uid` must hash to distinct per-slice
    /// keys; otherwise an event for slice B would overwrite slice A's state.
    #[test]
    fn slice_key_falls_back_to_ns_and_name_when_uid_missing() {
        let a = make_slice_ns(&["10.0.0.1"], 30000, true, "ns1", "a");
        let b = make_slice_ns(&["10.0.0.2"], 30000, true, "ns1", "b");
        assert_ne!(slice_key(&a), slice_key(&b));
        assert_eq!(slice_key(&a), "ns1/a");
    }

    #[test]
    fn slice_key_uses_uid_when_present() {
        let mut a = make_slice_ns(&["10.0.0.1"], 30000, true, "ns1", "a");
        a.metadata.uid = Some("uid-abc".into());
        assert_eq!(slice_key(&a), "uid-abc");
    }

    fn with_uid(mut es: EndpointSlice, uid: &str) -> EndpointSlice {
        es.metadata.uid = Some(uid.into());
        es
    }

    /// Apply → Delete on the same slice removes its workers from the union
    /// (the missing variant for `.applied_objects()` before the rewrite).
    #[tokio::test]
    async fn delete_event_emits_removed_for_workers_in_slice() {
        let s = with_uid(make_slice_ns(&["10.0.0.1"], 30000, true, "ns", "svc"), "u1");
        let events = vec![
            Ok(watcher::Event::Apply(s.clone())),
            Ok(watcher::Event::Delete(s)),
        ];
        let (tx, mut rx) = mpsc::channel(16);
        let stream = futures::stream::iter(events);
        process_events(stream, tx, plain_mode()).await;
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(matches!(out[0], DiscoveryEvent::Added(_)));
        assert!(matches!(out[1], DiscoveryEvent::Removed { .. }));
    }

    /// Init/InitDone replaces state atomically: any worker present before
    /// the Init cycle but not seen during it is diffed out as Removed.  This
    /// covers the "slice deleted while watcher was disconnected" case.
    #[tokio::test]
    async fn init_cycle_diffs_out_unseen_slices() {
        let a = with_uid(make_slice_ns(&["10.0.0.1"], 30000, true, "ns", "a"), "u-a");
        let b = with_uid(make_slice_ns(&["10.0.0.2"], 30000, true, "ns", "b"), "u-b");

        let events = vec![
            // Initial state: both slices live.
            Ok(watcher::Event::Apply(a.clone())),
            Ok(watcher::Event::Apply(b.clone())),
            // Watcher restart resyncs and only sees `a` (b was deleted offline).
            Ok(watcher::Event::Init),
            Ok(watcher::Event::InitApply(a.clone())),
            Ok(watcher::Event::InitDone),
        ];
        let (tx, mut rx) = mpsc::channel(16);
        process_events(futures::stream::iter(events), tx, plain_mode()).await;
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        let removed: Vec<_> = out
            .iter()
            .filter_map(|e| match e {
                DiscoveryEvent::Removed { id } => Some(id.0.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            removed.contains(&"ns/b/10.0.0.2:30000"),
            "init resync should diff out the deleted slice: out={out:?}"
        );
        assert!(
            !removed.contains(&"ns/a/10.0.0.1:30000"),
            "live slice must not be removed: out={out:?}"
        );
    }

    /// When the consumer drops the receiver, the watcher loop exits cleanly
    /// rather than continuing to process events into the void.
    #[tokio::test]
    async fn watcher_exits_when_consumer_drops_receiver() {
        use std::time::Duration;
        let s = with_uid(make_slice_ns(&["10.0.0.1"], 30000, true, "ns", "a"), "u-a");
        // A stream that never ends — only consumer drop should stop the loop.
        let events = futures::stream::iter(std::iter::repeat_with(move || {
            Ok(watcher::Event::Apply(s.clone()))
        }));
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let handle = tokio::spawn(process_events(events, tx, plain_mode()));
        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("process_events must exit promptly when consumer drops")
            .expect("task should not panic");
    }

    /// Watcher errors are logged but state is preserved; the next successful
    /// event continues to diff against the pre-error union.
    #[tokio::test]
    async fn watcher_error_preserves_state_and_diffs_against_pre_error_union() {
        let s = with_uid(make_slice_ns(&["10.0.0.1"], 30000, true, "ns", "a"), "u-a");
        let events = vec![
            Ok(watcher::Event::Apply(s.clone())),
            Err(watcher::Error::NoResourceVersion),
            // Same slice; nothing should be re-emitted because state survived.
            Ok(watcher::Event::Apply(s)),
        ];
        let (tx, mut rx) = mpsc::channel(16);
        process_events(futures::stream::iter(events), tx, plain_mode()).await;
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        // Exactly one Added — the second Apply is a no-op against the
        // existing union.
        let added = out
            .iter()
            .filter(|e| matches!(e, DiscoveryEvent::Added(_)))
            .count();
        assert_eq!(added, 1, "out={out:?}");
    }

    /// In PD mode, only slices whose labels match one of the role selectors
    /// produce workers; unrelated slices in the same namespace are dropped
    /// without registration.
    #[tokio::test]
    async fn pd_mode_drops_slices_whose_labels_match_no_role_selector() {
        let prefill_slice = with_uid(
            make_slice_full(
                &["10.0.0.1"],
                30000,
                true,
                "ns",
                "p",
                &[("app", "sglang"), ("role", "prefill")],
            ),
            "u-p",
        );
        let unrelated_slice = with_uid(
            make_slice_full(
                &["10.0.0.2"],
                30000,
                true,
                "ns",
                "x",
                &[("app", "sglang"), ("role", "router")],
            ),
            "u-x",
        );
        let events = vec![
            Ok(watcher::Event::Apply(prefill_slice)),
            Ok(watcher::Event::Apply(unrelated_slice)),
        ];
        let (tx, mut rx) = mpsc::channel(16);
        process_events(futures::stream::iter(events), tx, pd_mode()).await;
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        let added: Vec<_> = out
            .iter()
            .filter_map(|e| match e {
                DiscoveryEvent::Added(spec) => Some(spec),
                _ => None,
            })
            .collect();
        assert_eq!(
            added.len(),
            1,
            "only the prefill-labelled slice should be registered: out={out:?}"
        );
        assert_eq!(added[0].mode, WorkerMode::Prefill);
        assert_eq!(added[0].id.0, "ns/p/10.0.0.1:30000");
    }

    /// End-to-end K8s + PD integration: synthesize EndpointSlice events
    /// for two prefill pods and two decode pods (each backed by a real
    /// HTTP listener mounting `/server_info`), pipe them through
    /// `process_events` → DiscoveryEvent channel → manager, and assert
    /// the resulting registry has:
    ///   * two `Prefill` workers, each with `bootstrap_port` matching
    ///     what its own `/server_info` advertised (so per-worker
    ///     plumbing is verified, not just "some prefill registered"),
    ///   * two `Decode` workers with `bootstrap_port = None`.
    ///
    /// This is the load-bearing integration covering the seam this PR
    /// just opened: the K8s backend emits `bootstrap_port: None`, the
    /// PD-disaggregation classification comes from slice labels, and
    /// `WorkerMode` + `bootstrap_port` are re-resolved by the manager
    /// from each worker's `/server_info`. A regression at *any* of
    /// those three layers (k8s extract → process_events label
    /// classification → manager introspect-and-override) fails this
    /// test.
    #[tokio::test]
    async fn k8s_pd_pipeline_registers_workers_with_per_pod_bootstrap_port() {
        use crate::workers::introspect::WorkerIntrospector;
        use crate::workers::manager::run_with_introspector;
        use crate::workers::WorkerRegistry;
        use axum::{routing::get, Json, Router};
        use serde_json::{json, Value};
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::net::TcpListener;
        use tokio::sync::oneshot;

        /// Bind axum on an OS-assigned 127.0.0.1 port, mount a
        /// `/server_info` returning `body`, return the port + shutdown
        /// channel so the test can join cleanly.
        async fn spawn_fake_server_info(body: Value) -> (u16, oneshot::Sender<()>) {
            let body = Arc::new(body);
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let app = Router::new().route(
                "/server_info",
                get(move || {
                    let body = body.clone();
                    async move { Json((*body).clone()) }
                }),
            );
            let (tx, rx) = oneshot::channel::<()>();
            tokio::spawn(async move {
                let _ = axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        let _ = rx.await;
                    })
                    .await;
            });
            (port, tx)
        }

        // Four fake SGLang workers — two prefill (each with a distinct
        // bootstrap_port to verify per-pod plumbing) and two decode.
        let (port_p1, _shut_p1) = spawn_fake_server_info(json!({
            "served_model_name": "m",
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 8998,
        }))
        .await;
        let (port_p2, _shut_p2) = spawn_fake_server_info(json!({
            "served_model_name": "m",
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 8999,
        }))
        .await;
        let (port_d1, _shut_d1) = spawn_fake_server_info(json!({
            "served_model_name": "m",
            "disaggregation_mode": "decode",
        }))
        .await;
        let (port_d2, _shut_d2) = spawn_fake_server_info(json!({
            "served_model_name": "m",
            "disaggregation_mode": "decode",
        }))
        .await;

        // One EndpointSlice per worker (one address each, so the slice
        // port matches the worker's axum listener port exactly).
        // Labels match the PD selectors so `classify_mode` sends each
        // slice to the right pool.
        let prefill_labels = &[("app", "sglang"), ("role", "prefill")];
        let decode_labels = &[("app", "sglang"), ("role", "decode")];
        let p1 = with_uid(
            make_slice_full(
                &["127.0.0.1"],
                port_p1 as i32,
                true,
                "ns",
                "prefill-1",
                prefill_labels,
            ),
            "u-p1",
        );
        let p2 = with_uid(
            make_slice_full(
                &["127.0.0.1"],
                port_p2 as i32,
                true,
                "ns",
                "prefill-2",
                prefill_labels,
            ),
            "u-p2",
        );
        let d1 = with_uid(
            make_slice_full(
                &["127.0.0.1"],
                port_d1 as i32,
                true,
                "ns",
                "decode-1",
                decode_labels,
            ),
            "u-d1",
        );
        let d2 = with_uid(
            make_slice_full(
                &["127.0.0.1"],
                port_d2 as i32,
                true,
                "ns",
                "decode-2",
                decode_labels,
            ),
            "u-d2",
        );

        // Manager pipeline: DiscoveryEvent channel → run_with_introspector.
        let registry = Arc::new(WorkerRegistry::default());
        let (dtx, drx) = mpsc::channel::<DiscoveryEvent>(16);
        let introspector = Arc::new(WorkerIntrospector::new(Duration::from_millis(500)));
        let manager_handle = tokio::spawn(run_with_introspector(
            drx,
            registry.clone(),
            None,
            None,
            None,
            introspector,
        ));

        // Drive process_events with the four slice Apply events, then
        // drop dtx so the manager loop exits cleanly once it has drained.
        let events = vec![
            Ok(watcher::Event::Apply(p1)),
            Ok(watcher::Event::Apply(p2)),
            Ok(watcher::Event::Apply(d1)),
            Ok(watcher::Event::Apply(d2)),
        ];
        let producer = tokio::spawn(async move {
            let stream = futures::stream::iter(events);
            process_events(stream, dtx, pd_mode()).await;
        });

        // Poll the registry until all four workers are present with
        // their resolved mode + bootstrap_port — order isn't deterministic
        // because each worker's `/server_info` round-trip happens in a
        // separate manager task.
        let expected_p1_id = WorkerId(format!("ns/prefill-1/127.0.0.1:{port_p1}"));
        let expected_p2_id = WorkerId(format!("ns/prefill-2/127.0.0.1:{port_p2}"));
        let expected_d1_id = WorkerId(format!("ns/decode-1/127.0.0.1:{port_d1}"));
        let expected_d2_id = WorkerId(format!("ns/decode-2/127.0.0.1:{port_d2}"));

        let settled = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let p1 = registry.get(&expected_p1_id);
                let p2 = registry.get(&expected_p2_id);
                let d1 = registry.get(&expected_d1_id);
                let d2 = registry.get(&expected_d2_id);
                if let (Some(p1), Some(p2), Some(d1), Some(d2)) = (p1, p2, d1, d2) {
                    if p1.mode() == WorkerMode::Prefill
                        && p2.mode() == WorkerMode::Prefill
                        && d1.mode() == WorkerMode::Decode
                        && d2.mode() == WorkerMode::Decode
                        && p1.bootstrap_port() == Some(8998)
                        && p2.bootstrap_port() == Some(8999)
                        && d1.bootstrap_port().is_none()
                        && d2.bootstrap_port().is_none()
                    {
                        return true;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            settled.is_ok(),
            "registry did not converge to (2 prefill + 2 decode) with per-pod \
             bootstrap_port within 3s. current state: \
             p1={:?}, p2={:?}, d1={:?}, d2={:?}",
            registry
                .get(&expected_p1_id)
                .map(|w| (w.mode(), w.bootstrap_port())),
            registry
                .get(&expected_p2_id)
                .map(|w| (w.mode(), w.bootstrap_port())),
            registry
                .get(&expected_d1_id)
                .map(|w| (w.mode(), w.bootstrap_port())),
            registry
                .get(&expected_d2_id)
                .map(|w| (w.mode(), w.bootstrap_port())),
        );

        // Producer should exit when the iter stream ends; manager exits
        // when the producer drops dtx. Both should finish quickly.
        let _ = tokio::time::timeout(Duration::from_secs(1), producer).await;
        let _ = tokio::time::timeout(Duration::from_secs(1), manager_handle).await;
    }

    /// Helper: build a slice where every endpoint carries a synthetic
    /// `target_ref.uid`. The endpoint at position `i` gets `uids[i]`.
    fn make_slice_with_uids(addrs: &[&str], port: i32, uids: &[&str]) -> EndpointSlice {
        assert_eq!(addrs.len(), uids.len());
        let endpoints = addrs
            .iter()
            .zip(uids.iter())
            .map(|(addr, uid)| Endpoint {
                addresses: vec![(*addr).to_string()],
                conditions: Some(EndpointConditions {
                    ready: Some(true),
                    ..Default::default()
                }),
                target_ref: Some(ObjectReference {
                    uid: Some((*uid).to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .collect();
        EndpointSlice {
            metadata: ObjectMeta {
                name: Some("svc".into()),
                namespace: Some("ns".into()),
                ..Default::default()
            },
            address_type: "IPv4".into(),
            endpoints,
            ports: Some(vec![EndpointPort {
                port: Some(port),
                ..Default::default()
            }]),
        }
    }

    /// Pod is replaced (same IP, different UID) — router must see this as
    /// a Removed+Added cycle so the new pod gets fresh CB/router_inflight_load
    /// state. Without UID-keyed WorkerIds, two consecutive
    /// `process_events` snapshots would dedup by `addr:port` and the
    /// new pod would inherit the dead pod's state.
    #[tokio::test]
    async fn pod_replace_with_same_ip_emits_remove_then_add() {
        let s_old = with_uid(
            make_slice_with_uids(&["10.0.0.1"], 30000, &["uid-old"]),
            "u-1",
        );
        let s_new = with_uid(
            make_slice_with_uids(&["10.0.0.1"], 30000, &["uid-new"]),
            "u-1",
        );
        let (tx, mut rx) = mpsc::channel(16);
        process_events(
            futures::stream::iter(vec![
                Ok(watcher::Event::Apply(s_old)),
                Ok(watcher::Event::Apply(s_new)),
            ]),
            tx,
            plain_mode(),
        )
        .await;
        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        // Expect: Added(uid-old) → Removed(uid-old) + Added(uid-new).
        // The order of Removed/Added within the second apply depends on
        // emit_diff's iteration; assert by counting each variant.
        assert_eq!(events.len(), 3, "got {events:?}");
        let added: Vec<&WorkerSpec> = events
            .iter()
            .filter_map(|e| match e {
                DiscoveryEvent::Added(spec) => Some(spec),
                _ => None,
            })
            .collect();
        let removed: Vec<&WorkerId> = events
            .iter()
            .filter_map(|e| match e {
                DiscoveryEvent::Removed { id } => Some(id),
                _ => None,
            })
            .collect();
        assert_eq!(added.len(), 2, "two Added (one per UID): {events:?}");
        assert_eq!(removed.len(), 1, "one Removed (for uid-old): {events:?}");
        assert_eq!(
            added[0].id.0, "ns/uid-old",
            "first Added is the original pod",
        );
        assert_eq!(
            added[1].id.0, "ns/uid-new",
            "second Added is the replacement pod with a fresh UID",
        );
        assert_eq!(
            removed[0].0, "ns/uid-old",
            "Removed targets the original pod's UID, not the IP-keyed id",
        );
        // Same URL across both, confirming the IP didn't change.
        assert_eq!(added[0].url, added[1].url);
    }

    /// End-to-end reconcile: an EndpointSlice flips `ready=true` while the
    /// engine's `/server_info` is still failing (503) — the production
    /// race where K8s readiness (cheap `/health`) leads the
    /// scheduler-backed `/server_info`. The worker registers with empty
    /// `model_ids` (invisible to routing), and discovery emits no further
    /// event. The manager's reconcile loop must re-introspect it and move
    /// it into the model pool once `/server_info` recovers — driven
    /// through the real `process_events` → manager path.
    #[tokio::test]
    async fn k8s_reconcile_recovers_worker_whose_server_info_was_initially_failing() {
        use crate::discovery::ModelId;
        use crate::workers::introspect::WorkerIntrospector;
        use crate::workers::manager::run_with_introspector_and_reconcile;
        use crate::workers::WorkerRegistry;
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        use axum::{routing::get, Json, Router};
        use serde_json::json;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::net::TcpListener;

        // Fake engine: 503 on /server_info until `ready` flips true, then
        // serves a valid body advertising model "m".
        let ready = Arc::new(AtomicBool::new(false));
        let ready_handler = ready.clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new().route(
            "/server_info",
            get(move || {
                let ready = ready_handler.clone();
                async move {
                    if ready.load(Ordering::SeqCst) {
                        Json(json!({"served_model_name": "m"})).into_response()
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    }
                }
            }),
        );
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });

        // EndpointSlice already marked ready=true (kubelet's /health probe
        // passed) pointing at the engine whose /server_info is still 503.
        let slice = with_uid(
            make_slice_full(&["127.0.0.1"], port as i32, true, "ns", "engine-1", &[]),
            "u-e1",
        );

        let registry = Arc::new(WorkerRegistry::default());
        let (dtx, drx) = mpsc::channel::<DiscoveryEvent>(16);
        // Hold a second sender so the channel stays open after the
        // producer's single-event stream ends — otherwise the manager
        // loop would exit before any reconcile tick fires.
        let dtx_keepalive = dtx.clone();
        let introspector = Arc::new(WorkerIntrospector::new(Duration::from_millis(300)));
        let manager_handle = tokio::spawn(run_with_introspector_and_reconcile(
            drx,
            registry.clone(),
            None,
            None,
            None,
            introspector,
            Duration::from_millis(150),
        ));

        // Drive the ready=true slice through the real discovery processor.
        let producer = tokio::spawn(async move {
            let stream = futures::stream::iter(vec![Ok(watcher::Event::Apply(slice))]);
            process_events(stream, dtx, plain_mode()).await;
        });

        // No target_ref on the endpoint => id falls back to ns/slice/addr:port.
        let id = WorkerId(format!("ns/engine-1/127.0.0.1:{port}"));
        let model = ModelId("m".into());

        // Phase 1: worker is registered but absent from the model pool
        // while /server_info keeps failing.
        let stuck = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(w) = registry.get(&id) {
                    if w.model_ids.is_empty() && registry.workers_for(&model).is_empty() {
                        return true;
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            stuck.is_ok(),
            "worker should register with empty model_ids while /server_info is failing",
        );

        // Engine finishes coming up.
        ready.store(true, Ordering::SeqCst);

        // Phase 2: reconcile re-introspects and the worker joins the pool,
        // with no further discovery event.
        let recovered = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if !registry.workers_for(&model).is_empty() {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            recovered.is_ok(),
            "reconcile must re-introspect the worker and add it to the model pool once \
             /server_info recovers; registry state: {:?}",
            registry.get(&id).map(|w| w.model_ids.clone()),
        );

        drop(dtx_keepalive);
        let _ = shutdown_tx.send(());
        let _ = tokio::time::timeout(Duration::from_secs(1), producer).await;
        let _ = tokio::time::timeout(Duration::from_secs(1), manager_handle).await;
    }

    // -----------------------------------------------------------------------
    // Peer discovery — the sibling-replica set.
    // -----------------------------------------------------------------------

    /// Multiple endpoints in one slice, each with its own ready condition, so
    /// the per-endpoint filtering can be exercised.
    fn peer_slice(entries: &[(&str, Option<bool>)], port: Option<i32>) -> EndpointSlice {
        EndpointSlice {
            metadata: ObjectMeta {
                name: Some("sgl-router-kv-abc".into()),
                namespace: Some("sgl-router-test".into()),
                ..Default::default()
            },
            address_type: "IPv4".into(),
            endpoints: entries
                .iter()
                .map(|(addr, ready)| Endpoint {
                    addresses: vec![(*addr).to_string()],
                    conditions: ready.map(|r| EndpointConditions {
                        ready: Some(r),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
            ports: port.map(|p| {
                vec![EndpointPort {
                    port: Some(p),
                    ..Default::default()
                }]
            }),
        }
    }

    fn identity_with_ip(ip: &str) -> SelfIdentity {
        SelfIdentity {
            ip: Some(ip.into()),
            ..Default::default()
        }
    }

    fn identity_with_pod(name: &str, namespace: Option<&str>) -> SelfIdentity {
        SelfIdentity {
            pod_name: Some(name.into()),
            pod_namespace: namespace.map(Into::into),
            ..Default::default()
        }
    }

    /// One endpoint per (addr, pod-name) entry, each pointing its `target_ref`
    /// at the named pod in `sgl-router-test`.
    fn pod_backed_peer_slice(entries: &[(&str, &str)], port: Option<i32>) -> EndpointSlice {
        let mut es = peer_slice(
            &entries
                .iter()
                .map(|(addr, _)| (*addr, Some(true)))
                .collect::<Vec<_>>(),
            port,
        );
        for (ep, (_, pod)) in es.endpoints.iter_mut().zip(entries) {
            ep.target_ref = Some(ObjectReference {
                kind: Some("Pod".into()),
                name: Some((*pod).into()),
                namespace: Some("sgl-router-test".into()),
                uid: Some(format!("uid-{pod}")),
                ..Default::default()
            });
        }
        es
    }

    /// `PeerFilter::extract`, reduced to the URLs it offers.
    fn extracted_urls(
        es: &EndpointSlice,
        identity: SelfIdentity,
        family: AddressFamily,
        port: i32,
    ) -> Vec<String> {
        PeerFilter {
            identity,
            family,
            port,
        }
        .extract(es)
        .into_iter()
        .map(|p| p.url)
        .collect()
    }

    #[test]
    fn peer_address_family_follows_the_listen_socket() {
        assert_eq!(peer_address_family("0.0.0.0"), AddressFamily::V4);
        assert_eq!(peer_address_family("10.0.0.1"), AddressFamily::V4);
        assert_eq!(peer_address_family("::"), AddressFamily::Any);
        assert_eq!(peer_address_family("fd00::1"), AddressFamily::V6);
        assert_eq!(peer_address_family("router.local"), AddressFamily::Any);
    }

    #[test]
    fn extract_peers_excludes_self() {
        let es = peer_slice(
            &[("10.0.0.1", Some(true)), ("10.0.0.2", Some(true))],
            Some(8090),
        );
        assert_eq!(
            extracted_urls(&es, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090),
            vec!["http://10.0.0.2:8090"],
            "a replica must not list itself as a peer",
        );
    }

    /// POD_NAME matches the endpoint's `target_ref` and excludes the whole
    /// pod, even when the endpoint's address is not POD_IP — which is what a
    /// dual-stack Service's secondary family looks like.
    #[test]
    fn extract_peers_excludes_self_by_pod_name() {
        let es = pod_backed_peer_slice(
            &[
                ("10.0.0.1", "sgl-router-kv-0"),
                ("10.0.0.2", "sgl-router-kv-1"),
            ],
            Some(8090),
        );
        let identity = SelfIdentity {
            ip: Some("192.168.1.5".into()), // primary addr, NOT in this slice
            ..identity_with_pod("sgl-router-kv-0", Some("sgl-router-test"))
        };
        assert_eq!(
            extracted_urls(&es, identity, AddressFamily::V4, 8090),
            vec!["http://10.0.0.2:8090"],
            "target_ref matching must exclude the pod even at an unknown address",
        );
    }

    /// With POD_NAMESPACE known, a same-named pod in another namespace is a
    /// different pod and must be kept.
    #[test]
    fn extract_peers_keeps_same_named_pod_in_another_namespace() {
        let mut es = pod_backed_peer_slice(&[("10.0.0.2", "sgl-router-kv-0")], Some(8090));
        es.endpoints[0].target_ref.as_mut().unwrap().namespace = Some("other-ns".into());
        assert_eq!(
            extracted_urls(
                &es,
                identity_with_pod("sgl-router-kv-0", Some("sgl-router-test")),
                AddressFamily::V4,
                8090
            ),
            vec!["http://10.0.0.2:8090"],
        );
    }

    /// Without POD_NAMESPACE a bare name match still excludes — EndpointSlices
    /// are namespaced, so a collision needs the same Deployment name twice.
    #[test]
    fn extract_peers_excludes_by_name_when_own_namespace_unknown() {
        let es = pod_backed_peer_slice(&[("10.0.0.1", "sgl-router-kv-0")], Some(8090));
        assert!(extracted_urls(
            &es,
            identity_with_pod("sgl-router-kv-0", None),
            AddressFamily::V4,
            8090
        )
        .is_empty());
    }

    /// Endpoints with no `target_ref` (manually-written slices) still fall
    /// back to POD_IP matching.
    #[test]
    fn extract_peers_falls_back_to_ip_without_target_ref() {
        let es = peer_slice(&[("10.0.0.1", Some(true))], Some(8090));
        let identity = SelfIdentity {
            ip: Some("10.0.0.1".into()),
            ..identity_with_pod("sgl-router-kv-0", Some("sgl-router-test"))
        };
        assert!(extracted_urls(&es, identity, AddressFamily::V4, 8090).is_empty());
    }

    /// An endpoint's addresses are fungible — one backend — so a self address
    /// anywhere in the list makes the whole endpoint this replica.
    #[test]
    fn extract_peers_ip_fallback_excludes_the_whole_endpoint() {
        let mut es = peer_slice(&[("10.0.0.9", Some(true))], Some(8090));
        es.endpoints[0].addresses = vec!["10.0.0.9".into(), "10.0.0.1".into()];
        assert!(
            extracted_urls(&es, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090).is_empty(),
            "a self address must exclude its endpoint, not just itself",
        );
    }

    /// For the same reason a sibling endpoint is one peer, at its first
    /// address, however many addresses it lists.
    #[test]
    fn extract_peers_offers_one_url_per_endpoint() {
        let mut es = peer_slice(&[("10.0.0.2", Some(true))], Some(8090));
        es.endpoints[0].addresses = vec!["10.0.0.2".into(), "10.0.0.3".into()];
        assert_eq!(
            extracted_urls(&es, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090),
            vec!["http://10.0.0.2:8090"],
        );
    }

    /// A `target_ref` pointing at a non-Pod object (a hand-written slice
    /// naming a Service, say) must not be mistaken for this replica.
    #[test]
    fn extract_peers_ignores_target_ref_of_non_pod_kind() {
        let mut es = pod_backed_peer_slice(&[("10.0.0.2", "sgl-router-kv-0")], Some(8090));
        es.endpoints[0].target_ref.as_mut().unwrap().kind = Some("Service".into());
        assert_eq!(
            extracted_urls(
                &es,
                identity_with_pod("sgl-router-kv-0", Some("sgl-router-test")),
                AddressFamily::V4,
                8090
            ),
            vec!["http://10.0.0.2:8090"],
        );
    }

    /// A `target_ref` with no kind (hand-written slice) is treated as a Pod:
    /// excluding beats listing this replica as its own peer.
    #[test]
    fn extract_peers_treats_absent_target_ref_kind_as_pod() {
        let mut es = pod_backed_peer_slice(&[("10.0.0.1", "sgl-router-kv-0")], Some(8090));
        es.endpoints[0].target_ref.as_mut().unwrap().kind = None;
        assert!(extracted_urls(
            &es,
            identity_with_pod("sgl-router-kv-0", Some("sgl-router-test")),
            AddressFamily::V4,
            8090
        )
        .is_empty());
    }

    /// A set-but-empty env var must behave as unset; otherwise it would
    /// suppress the `is_unknown` warning while matching nothing.
    #[test]
    fn env_non_empty_treats_empty_strings_as_unset() {
        // Unique names so parallel tests cannot race on these vars.
        std::env::set_var("SGL_ROUTER_TEST_ENV_EMPTY", "");
        std::env::set_var("SGL_ROUTER_TEST_ENV_SET", "x");
        assert_eq!(env_non_empty("SGL_ROUTER_TEST_ENV_EMPTY"), None);
        assert_eq!(
            env_non_empty("SGL_ROUTER_TEST_ENV_SET"),
            Some("x".to_string())
        );
        assert_eq!(env_non_empty("SGL_ROUTER_TEST_ENV_MISSING"), None);
        std::env::remove_var("SGL_ROUTER_TEST_ENV_EMPTY");
        std::env::remove_var("SGL_ROUTER_TEST_ENV_SET");
    }

    /// With neither POD_NAME nor POD_IP there is nothing to exclude, so this
    /// replica stays in its own peer set.
    #[test]
    fn extract_peers_keeps_self_when_identity_is_unknown() {
        let es = peer_slice(&[("10.0.0.1", Some(true))], Some(8090));
        assert_eq!(
            extracted_urls(&es, SelfIdentity::default(), AddressFamily::V4, 8090),
            vec!["http://10.0.0.1:8090"],
        );
        assert!(SelfIdentity::default().is_unknown());
    }

    /// Per the EndpointSlice API an ABSENT ready condition means ready.
    /// Treating it as not-ready would empty the peer set on clusters that omit
    /// it.
    #[test]
    fn extract_peers_treats_absent_ready_as_ready() {
        let es = peer_slice(&[("10.0.0.2", None)], Some(8090));
        assert_eq!(
            extracted_urls(&es, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090),
            vec!["http://10.0.0.2:8090"],
        );
    }

    /// An unready sibling is either still starting or draining, so it is left
    /// out of the peer set; in a rolling update, a new replica does not list
    /// another still-starting one.
    #[test]
    fn extract_peers_skips_unready_endpoints() {
        let es = peer_slice(
            &[("10.0.0.2", Some(false)), ("10.0.0.3", Some(true))],
            Some(8090),
        );
        assert_eq!(
            extracted_urls(&es, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090),
            vec!["http://10.0.0.3:8090"],
        );
    }

    #[test]
    fn extract_peers_brackets_ipv6_addresses() {
        let mut es = peer_slice(&[("fd00::2", Some(true))], Some(8090));
        es.address_type = "IPv6".into();
        assert_eq!(
            extracted_urls(&es, identity_with_ip("fd00::1"), AddressFamily::V6, 8090),
            vec!["http://[fd00::2]:8090"],
        );
    }

    /// A dual-stack Service yields one slice per family; only the family this
    /// router listens on is kept.
    #[test]
    fn extract_peers_skips_the_other_address_family() {
        let mut v6 = peer_slice(&[("fd00::2", Some(true))], Some(8090));
        v6.address_type = "IPv6".into();
        assert!(
            extracted_urls(&v6, identity_with_ip("10.0.0.1"), AddressFamily::V4, 8090).is_empty(),
            "an IPv4 router must ignore the IPv6 slice",
        );

        let v4 = peer_slice(&[("10.0.0.2", Some(true))], Some(8090));
        assert!(
            extracted_urls(&v4, identity_with_ip("fd00::1"), AddressFamily::V6, 8090).is_empty(),
            "an IPv6 router must ignore the IPv4 slice",
        );
    }

    /// A listener that names no family (`::`, the dual-stack listener) keeps
    /// both slices.
    #[test]
    fn an_unspecified_listener_keeps_both_address_families() {
        let v4 = peer_slice(&[("10.0.0.2", Some(true))], Some(8090));
        let mut v6 = peer_slice(&[("fd00::2", Some(true))], Some(8090));
        v6.address_type = "IPv6".into();

        assert_eq!(
            extracted_urls(&v4, SelfIdentity::default(), AddressFamily::Any, 8090),
            vec!["http://10.0.0.2:8090"],
        );
        assert_eq!(
            extracted_urls(&v6, SelfIdentity::default(), AddressFamily::Any, 8090),
            vec!["http://[fd00::2]:8090"],
        );
    }

    /// A multi-port Service lists its ports in spec order, and nothing says
    /// the one this router serves on sorts first. Siblings are this router's
    /// own pods, so its own listen port is the reliable signal — taking index
    /// 0 would address every sibling on, say, a `metrics` port.
    #[test]
    fn peer_port_prefers_this_replicas_own_listen_port() {
        let mut es = peer_slice(&[("10.0.0.2", Some(true))], None);
        es.ports = Some(vec![
            EndpointPort {
                name: Some("metrics".into()),
                port: Some(9090),
                ..Default::default()
            },
            EndpointPort {
                name: Some("http".into()),
                port: Some(8090),
                ..Default::default()
            },
        ]);
        assert_eq!(
            extracted_urls(&es, SelfIdentity::default(), AddressFamily::Any, 8090),
            vec!["http://10.0.0.2:8090"],
            "the router's own port wins over whatever the Service lists first",
        );
    }

    /// When no advertised port matches — a Service fronting the router on a
    /// different port than it listens on — fall back to the first rather than
    /// to the hard-coded default, which would be wrong more often.
    #[test]
    fn peer_port_falls_back_to_the_first_advertised_port() {
        let es = peer_slice(&[("10.0.0.2", Some(true))], Some(8080));
        assert_eq!(
            extracted_urls(&es, SelfIdentity::default(), AddressFamily::Any, 8090),
            vec!["http://10.0.0.2:8080"],
        );
    }

    /// An FQDN slice cannot be classified by family, so it is kept.
    #[test]
    fn extract_peers_keeps_fqdn_slices_for_either_family() {
        let mut es = peer_slice(&[("router-1.svc", Some(true))], Some(8090));
        es.address_type = "FQDN".into();
        assert_eq!(
            extracted_urls(&es, SelfIdentity::default(), AddressFamily::V4, 8090),
            vec!["http://router-1.svc:8090"],
        );
    }

    #[test]
    fn extract_peers_falls_back_to_own_port_when_the_slice_lists_none() {
        let es = peer_slice(&[("10.0.0.2", Some(true))], None);
        assert_eq!(
            extracted_urls(&es, SelfIdentity::default(), AddressFamily::V4, 8090),
            vec!["http://10.0.0.2:8090"],
            "siblings listen where this replica does, not on the compiled-in default",
        );
    }

    #[test]
    fn extract_peers_on_empty_slice_is_empty() {
        assert!(extracted_urls(
            &peer_slice(&[], Some(8090)),
            SelfIdentity::default(),
            AddressFamily::V4,
            8090
        )
        .is_empty());
    }

    /// Same slice shape, but with a caller-chosen name so `slice_key` yields
    /// distinct keys — needed to model a multi-slice Service.
    fn named_peer_slice(
        name: &str,
        entries: &[(&str, Option<bool>)],
        port: Option<i32>,
    ) -> EndpointSlice {
        let mut es = peer_slice(entries, port);
        es.metadata.name = Some(name.into());
        es
    }

    async fn run_peer_events(
        events: Vec<Result<watcher::Event<EndpointSlice>, watcher::Error>>,
        peers: &PeerRegistry,
        identity: SelfIdentity,
        family: AddressFamily,
    ) {
        let stream = futures::stream::iter(events);
        tokio::pin!(stream);
        let filter = PeerFilter {
            identity,
            family,
            port: 8090,
        };
        process_peer_events(stream, peers, &filter).await;
    }

    /// A partial relist must never make `known_to_have_no_peers()` true.
    ///
    /// Asserted mid-relist: the settled state after `InitDone` is correct even
    /// when a partial set was briefly published.
    #[tokio::test]
    async fn peer_relist_does_not_publish_a_partial_set() {
        let peers = PeerRegistry::new();
        // Init plus one self-only slice, and NO InitDone: nothing may be
        // published yet, so the registry must not even be marked synced.
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.1", Some(true))], // self — yields zero peers
                    Some(8090),
                ))),
            ],
            &peers,
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;

        assert!(
            !peers.synced(),
            "an incomplete relist must not mark the peer set synced",
        );
        assert!(
            !peers.known_to_have_no_peers(),
            "a partial relist must never read as 'this replica is alone'",
        );
    }

    /// And the completed relist publishes the full set.
    #[tokio::test]
    async fn peer_relist_publishes_once_complete() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.1", Some(true))], // self
                    Some(8090),
                ))),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-b",
                    &[("10.0.0.2", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
            ],
            &peers,
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;

        assert_eq!(peers.candidates(), vec!["http://10.0.0.2:8090"]);
        assert!(!peers.known_to_have_no_peers());
    }

    /// A relist drops slices that disappeared while the watch was down.
    #[tokio::test]
    async fn peer_relist_replaces_rather_than_merges() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.2", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
                // Watch restarts; slice-a is gone, slice-b appears.
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-b",
                    &[("10.0.0.3", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
            ],
            &peers,
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;
        assert_eq!(
            peers.candidates(),
            vec!["http://10.0.0.3:8090"],
            "a relist must replace the view, not merge into it",
        );
    }

    /// Steady-state add and remove of one slice out of several.
    #[tokio::test]
    async fn peer_apply_and_delete_update_incrementally() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.2", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
                Ok(watcher::Event::Apply(named_peer_slice(
                    "slice-b",
                    &[("10.0.0.3", Some(true))],
                    Some(8090),
                ))),
            ],
            &peers,
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;
        let mut got = peers.candidates();
        got.sort();
        assert_eq!(got, vec!["http://10.0.0.2:8090", "http://10.0.0.3:8090"]);

        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.2", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-b",
                    &[("10.0.0.3", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
                Ok(watcher::Event::Delete(named_peer_slice(
                    "slice-b",
                    &[("10.0.0.3", Some(true))],
                    Some(8090),
                ))),
            ],
            &peers,
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;
        assert_eq!(
            peers.candidates(),
            vec!["http://10.0.0.2:8090"],
            "deleting a slice drops only its own endpoints",
        );
    }

    /// A transient watcher error must preserve the peer set, not clear or
    /// republish it: an empty publish would make `known_to_have_no_peers()`
    /// true.
    #[tokio::test]
    async fn peer_watcher_error_preserves_the_peer_set() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.2", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
                Err(watcher::Error::NoResourceVersion),
            ],
            &peers,
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;

        assert_eq!(
            peers.candidates(),
            vec!["http://10.0.0.2:8090"],
            "a transient error must not disturb the peer set",
        );
        assert!(!peers.known_to_have_no_peers());
    }

    /// A genuinely single-replica deployment reads as alone:
    /// `known_to_have_no_peers()` is true once synced.
    #[tokio::test]
    async fn peer_relist_with_only_self_is_conclusively_alone() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(named_peer_slice(
                    "slice-a",
                    &[("10.0.0.1", Some(true))],
                    Some(8090),
                ))),
                Ok(watcher::Event::InitDone),
            ],
            &peers,
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;
        assert!(peers.is_empty());
        assert!(
            peers.known_to_have_no_peers(),
            "synced with only self ⇒ genuinely alone",
        );
    }

    /// A dual-stack Service lists each sibling once per family. With a
    /// listener that names no family both slices are kept, and the sibling
    /// must still be offered once, not once per family.
    #[tokio::test]
    async fn a_dual_stack_sibling_is_offered_once() {
        let v4 = pod_backed_peer_slice(&[("10.0.0.2", "sgl-router-kv-1")], Some(8090));
        let mut v6 = pod_backed_peer_slice(&[("fd00::2", "sgl-router-kv-1")], Some(8090));
        v6.metadata.name = Some("sgl-router-kv-v6".into());
        v6.address_type = "IPv6".into();
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![
                Ok(watcher::Event::Init),
                Ok(watcher::Event::InitApply(v4)),
                Ok(watcher::Event::InitApply(v6)),
                Ok(watcher::Event::InitDone),
            ],
            &peers,
            SelfIdentity::default(),
            AddressFamily::Any,
        )
        .await;
        assert_eq!(peers.candidates(), vec!["http://10.0.0.2:8090"]);
    }

    /// An `InitDone` outside an `Init` cycle must not publish: the empty
    /// relist buffer would mark the registry synced with zero peers.
    #[tokio::test]
    async fn a_stray_init_done_publishes_nothing() {
        let peers = PeerRegistry::new();
        run_peer_events(
            vec![Ok(watcher::Event::InitDone)],
            &peers,
            identity_with_ip("10.0.0.1"),
            AddressFamily::V4,
        )
        .await;
        assert!(!peers.synced());
        assert!(!peers.known_to_have_no_peers());
    }
}
