//! Verified erase: remove a memory, everything derived from it, and every
//! copy of its bytes the database holds, then prove it with a receipt.
//!
//! [`MenteDb::forget`] removes a memory from the indexes, the graph and the
//! record table. That is enough for recall, but the bytes stay behind: in the
//! freed record extent until it is reused, in the WAL until a checkpoint, as
//! an HNSW tombstone that keeps the vector for routing, in index snapshots,
//! in caches assembled from it, and in the `.v1` copy a migration keeps. And
//! the facts, consolidations, entity and community nodes derived from the
//! memory keep its content in their own words. [`MenteDb::erase`] covers all
//! of that; see `docs/FORGETTING.md` for the full list and the limits.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

use mentedb_core::MemoryNode;
use mentedb_core::edge::EdgeType;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;
use mentedb_storage::legacy::{
    LEGACY_KEPT_SUFFIX, LEGACY_MAP_FILE, LEGACY_PAGE_FILE, LEGACY_WAL_FILE,
};
use mentedb_storage::record::{DATA_FILE, WAL_FILE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::MenteDb;

/// How far an erase reaches and what it checks. The defaults erase the
/// memory and its descendants, run every scrub step and verify.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EraseOptions {
    /// Also erase the memories this one was derived from (the targets of its
    /// `Derived` edges, followed upward), and everything derived from those.
    /// Off: they are kept and listed in [`ErasureReceipt::ancestors_kept`].
    pub include_sources: bool,
    /// Scan every file in the database directory for the erased content and
    /// embedding bytes after the scrub, and report what it finds.
    pub verify: bool,
    /// Zero every free extent of the record file, not only the erased
    /// records' own. Free space holds forgotten records and the old copies
    /// of records that moved when they grew. Writers wait while it runs.
    pub scrub_free_space: bool,
    /// Overwrite and delete the `.v1` files a format migration keeps (a full
    /// copy of the database as it was before the migration).
    pub remove_legacy_copies: bool,
    /// Overwrite with zeros the old inode of every file the erase replaces
    /// (index and graph snapshots, caches), once nothing names it, so the
    /// bytes do not stay in the freed blocks. Files with another hard link
    /// are left alone. Unix only.
    pub zero_replaced_files: bool,
    /// Refuse an erase that would remove more memories than this, before
    /// anything is changed. Guards against a cascade through a hub node.
    pub max_cascade: usize,
    /// Cached text not linked to a memory id (trajectory topics, phantom
    /// gaps, speculative contexts) is dropped when it contains, or is
    /// contained in, an erased memory's content. Text shorter than this many
    /// bytes is not matched, so short words do not wipe unrelated entries.
    pub text_match_min_bytes: usize,
    /// Content shorter than this many bytes is not searched for by the
    /// verification scan (it would match unrelated data); its embedding
    /// still is. Listed in [`Verification::content_not_checked`].
    pub verify_min_content_bytes: usize,
    /// Read size of the verification scan.
    pub verify_chunk_bytes: usize,
    /// Test support: stop with an error right after the records are erased,
    /// as a crash there would, leaving the rest to the next open.
    #[doc(hidden)]
    #[serde(skip)]
    pub fault_after_records: bool,
}

impl Default for EraseOptions {
    fn default() -> Self {
        Self {
            include_sources: false,
            verify: true,
            scrub_free_space: true,
            remove_legacy_copies: true,
            zero_replaced_files: true,
            max_cascade: 10_000,
            text_match_min_bytes: 16,
            verify_min_content_bytes: 16,
            verify_chunk_bytes: 1 << 20,
            fault_after_records: false,
        }
    }
}

/// Why a memory was erased.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EraseReason {
    /// The memory the caller named.
    Requested,
    /// Derived from an erased memory (a fact from a turn, a consolidation,
    /// a dedup keeper, an entity or community node).
    Derived,
    /// A source of the requested memory, erased by `include_sources`.
    Source,
    /// A linked memory with byte identical content (an exact duplicate).
    Identical,
}

/// One erased memory in a receipt. Hashes, never the content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasedMemory {
    pub id: MemoryId,
    pub reason: EraseReason,
    /// SHA-256 of the content's UTF-8 bytes, lowercase hex.
    pub content_sha256: String,
    /// SHA-256 of the embedding as little endian f32 bytes, lowercase hex;
    /// None when the memory had no embedding.
    pub embedding_sha256: Option<String>,
}

/// Which scrub steps ran and what each one did.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScrubSteps {
    /// Records removed from the record file with their extents zeroed.
    pub records_zeroed: usize,
    /// HNSW nodes whose stored vector was overwritten (current and old
    /// versions of each erased id).
    pub index_vectors_scrubbed: usize,
    /// Bytes of record file free space zeroed; None when skipped.
    pub free_space_bytes_zeroed: Option<u64>,
    /// The WAL was checkpointed and its replaced file zeroed.
    pub wal_scrubbed: bool,
    /// Index, graph and cache snapshots were rewritten without the erased
    /// memories, and the graph edge log zeroed before truncation.
    pub snapshots_rewritten: bool,
    /// Old file versions zeroed after a snapshot replaced them.
    pub replaced_files_zeroed: usize,
    /// `.v1` legacy files overwritten and deleted.
    pub legacy_files_removed: Vec<String>,
    /// Cached entries dropped from the speculative cache, pain registry,
    /// trajectory, transition map and phantom tracker.
    pub cache_entries_dropped: usize,
}

/// A file in which the verification scan found erased bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualHit {
    /// Path relative to the database directory.
    pub file: String,
    pub memory: MemoryId,
    /// `content` or `embedding`.
    pub kind: String,
}

/// Result of scanning the database directory after an erase.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Verification {
    pub files_scanned: usize,
    pub bytes_scanned: u64,
    /// True when any erased content or embedding was found anywhere.
    pub residual_found: bool,
    pub hits: Vec<ResidualHit>,
    /// Files that could not be read (so were not checked).
    pub unreadable: Vec<String>,
    /// Memories whose content was too short to search for; their embedding
    /// was still searched.
    pub content_not_checked: Vec<MemoryId>,
}

/// What an erase removed and how, safe to store or show: it carries hashes
/// and ids, never content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureReceipt {
    pub requested: MemoryId,
    pub erased: Vec<ErasedMemory>,
    /// Memories the requested one was derived from that were kept (erase
    /// with `include_sources` to remove them). They may still hold the
    /// content the erased memory was made from.
    pub ancestors_kept: Vec<MemoryId>,
    /// User profile memories of the erased memories' owners. A profile is a
    /// summary written across many memories with no link to each, so it may
    /// still paraphrase what was erased; review or rebuild it.
    pub profiles_to_review: Vec<MemoryId>,
    /// When the erase finished, microseconds since the Unix epoch.
    pub erased_at_us: u64,
    pub steps: ScrubSteps,
    /// Present when the erase ran with `verify`.
    pub verification: Option<Verification>,
}

/// Lowercase hex SHA-256.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn embedding_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// True when `text` and `content` overlap enough to say the text came from
/// the content: one contains the other and the shorter is at least `min`
/// bytes.
fn text_matches(text: &str, content: &str, min: usize) -> bool {
    let (short, long) = if text.len() <= content.len() {
        (text, content)
    } else {
        (content, text)
    };
    short.len() >= min && long.contains(short)
}

impl MenteDb {
    /// Erase a memory and everything derived from it, overwrite every copy
    /// of their bytes the database holds, and return a receipt.
    ///
    /// The cascade follows `Derived` edges toward what was derived (a fact
    /// from a turn, a consolidation from its sources, a dedup keeper from
    /// the duplicate it absorbed, an entity or community node from its
    /// members), transitively. Shared entity and community nodes are erased
    /// too, even though they also summarize other memories: enrichment
    /// rebuilds them from the members that remain. Ancestors are kept unless
    /// [`EraseOptions::include_sources`] is set; either way an ancestor with
    /// byte identical content (an exact duplicate) is erased.
    ///
    /// Then the scrub: each record is freed and zeroed in one logged
    /// operation (replayed after a crash, so a memory never comes back),
    /// every HNSW vector of the ids is overwritten, BM25 postings and tags
    /// only they carried go, the caches built from them are cleared, the
    /// record file's free space is zeroed, the WAL is checkpointed and its
    /// old file zeroed, every snapshot is rewritten (old versions zeroed),
    /// and the `.v1` legacy copies are removed. With
    /// [`EraseOptions::verify`], every file in the directory is scanned for
    /// the erased content and embedding bytes.
    ///
    /// Erases run one at a time; other reads and writes proceed (each
    /// subsystem takes its own lock, in the same order as forget), except
    /// during the free space scrub, which holds the storage lock. A memory
    /// derived from an erased one while the erase runs is not part of its
    /// cascade, so run it when no enrichment is in flight for the account,
    /// or erase again.
    pub fn erase(&self, id: MemoryId, opts: EraseOptions) -> MenteResult<ErasureReceipt> {
        let _one_at_a_time = self.erase_serial.lock();
        info!(%id, include_sources = opts.include_sources, "erasing memory");
        if self.page_of(id).is_none() {
            return Err(MenteError::MemoryNotFound(id));
        }
        let plan = self.erase_plan(id, &opts)?;

        // Old inodes of files about to be replaced, captured before any of
        // them is rewritten.
        let replaced = if opts.zero_replaced_files {
            ReplacedFiles::capture(&self.path)
        } else {
            ReplacedFiles::default()
        };

        let mut steps = ScrubSteps::default();
        let nodes: Vec<&MemoryNode> = plan.nodes.values().collect();

        // Caches first, and their snapshots written now: they are derived,
        // so a crash from here on loses nothing, and the text they hold
        // (which recovery could not match without the content) is gone
        // before the records are.
        steps.cache_entries_dropped = self.drop_cached_text(&nodes, &opts);
        self.save_cognitive_snapshots();

        // From here a crash leaves records half erased. The marker makes the
        // next open finish the scrub (see `finish_pending_erase`).
        let finish = PendingErase {
            scrub_free_space: opts.scrub_free_space,
            remove_legacy_copies: opts.remove_legacy_copies,
        };
        finish.write(&self.path)?;

        let erasing: HashSet<MemoryId> = plan.order.iter().map(|(m, _)| *m).collect();
        let mut erased = Vec::with_capacity(plan.order.len());
        let mut owners = HashSet::new();
        // Derived memories first, the requested one last: a crash part way
        // leaves the requested memory in place, so erasing it again finds and
        // finishes the rest of the cascade.
        for (mid, reason) in plan.order.iter().rev() {
            let Some(node) = plan.nodes.get(mid) else {
                // A graph node with no stored memory: only the graph holds it.
                self.graph.remove_memory(*mid);
                continue;
            };
            steps.index_vectors_scrubbed += self.erase_one(node, &erasing)?;
            steps.records_zeroed += 1;
            owners.insert((node.user_id, node.agent_id));
            erased.push(ErasedMemory {
                id: *mid,
                reason: *reason,
                content_sha256: sha256_hex(node.content.as_bytes()),
                embedding_sha256: (!node.embedding.is_empty())
                    .then(|| sha256_hex(&embedding_bytes(&node.embedding))),
            });
        }
        erased.reverse();
        if opts.fault_after_records {
            return Err(MenteError::Storage(
                "erase stopped after the records (fault injected)".into(),
            ));
        }

        let finished = self.finish_erase(&finish)?;
        steps.free_space_bytes_zeroed = finished.free_space_bytes_zeroed;
        steps.wal_scrubbed = true;
        steps.snapshots_rewritten = true;
        steps.legacy_files_removed = finished.legacy_files_removed;
        steps.replaced_files_zeroed = replaced.zero_replaced();

        // From the tag index, not a scan of every memory.
        let profiles_to_review = self
            .memory_ids_with_tag("user_profile")
            .into_iter()
            .filter_map(|pid| self.get_memory(pid).ok())
            .filter(|p| owners.contains(&(p.user_id, p.agent_id)))
            .map(|p| p.id)
            .collect();

        let verification = opts
            .verify
            .then(|| scan_directory(&self.path, &nodes, &opts));
        if let Some(v) = &verification
            && v.residual_found
        {
            warn!(%id, hits = v.hits.len(), "erase verification found residual bytes");
        }
        let receipt = ErasureReceipt {
            requested: id,
            erased,
            ancestors_kept: plan.ancestors_kept,
            profiles_to_review,
            erased_at_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            steps,
            verification,
        };
        info!(
            %id,
            erased = receipt.erased.len(),
            residual = receipt.verification.as_ref().map(|v| v.residual_found),
            "erase complete"
        );
        Ok(receipt)
    }

    /// Decide what to erase, before anything changes. Reads the graph under
    /// its read lock only (never with the page map held), then loads each
    /// memory.
    fn erase_plan(&self, id: MemoryId, opts: &EraseOptions) -> MenteResult<ErasePlan> {
        let mut order: Vec<(MemoryId, EraseReason)> = vec![(id, EraseReason::Requested)];
        let mut chosen: HashSet<MemoryId> = HashSet::from([id]);
        let ancestors: Vec<MemoryId>;
        {
            let graph = self.graph.graph();
            let derived_from = |m: MemoryId| -> Vec<MemoryId> {
                graph
                    .outgoing(m)
                    .into_iter()
                    .filter(|(_, e)| e.edge_type == EdgeType::Derived)
                    .map(|(t, _)| t)
                    .collect()
            };
            let derived_into = |m: MemoryId| -> Vec<MemoryId> {
                graph
                    .incoming(m)
                    .into_iter()
                    .filter(|(_, e)| e.edge_type == EdgeType::Derived)
                    .map(|(s, _)| s)
                    .collect()
            };

            // Ancestors: only up from the requested memory, never up from
            // the descendants the cascade pulls in (an entity node's other
            // members are not this memory's sources).
            ancestors = bfs(id, &derived_from);
            if opts.include_sources {
                for a in &ancestors {
                    if chosen.insert(*a) {
                        order.push((*a, EraseReason::Source));
                    }
                }
            }
            let roots: Vec<MemoryId> = order.iter().map(|(m, _)| *m).collect();
            for r in roots {
                for d in bfs(r, &derived_into) {
                    if chosen.insert(d) {
                        order.push((d, EraseReason::Derived));
                    }
                }
            }
            if order.len() > opts.max_cascade {
                return Err(MenteError::CapacityExceeded(format!(
                    "erase of {id} would remove {} memories (max_cascade {})",
                    order.len(),
                    opts.max_cascade
                )));
            }
        }

        let mut nodes: HashMap<MemoryId, MemoryNode> = HashMap::new();
        for (m, _) in &order {
            if let Ok(n) = self.get_memory(*m) {
                nodes.insert(*m, n);
            }
        }

        // An ancestor with byte identical content is the same content, not
        // a source of it (exact dedup links the keeper to the duplicate it
        // absorbed): erase it and its descendants too.
        let mut ancestors_kept = Vec::new();
        let hashes: HashSet<String> = nodes
            .values()
            .map(|n| sha256_hex(n.content.as_bytes()))
            .collect();
        for a in ancestors {
            if chosen.contains(&a) {
                continue;
            }
            match self.get_memory(a) {
                Ok(n) if hashes.contains(&sha256_hex(n.content.as_bytes())) => {
                    chosen.insert(a);
                    order.push((a, EraseReason::Identical));
                    let more = {
                        let graph = self.graph.graph();
                        bfs(a, &|m| {
                            graph
                                .incoming(m)
                                .into_iter()
                                .filter(|(_, e)| e.edge_type == EdgeType::Derived)
                                .map(|(s, _)| s)
                                .collect()
                        })
                    };
                    nodes.insert(a, n);
                    for d in more {
                        if chosen.insert(d) {
                            order.push((d, EraseReason::Derived));
                            if let Ok(dn) = self.get_memory(d) {
                                nodes.insert(d, dn);
                            }
                        }
                    }
                }
                Ok(_) => ancestors_kept.push(a),
                Err(_) => {}
            }
        }
        if order.len() > opts.max_cascade {
            return Err(MenteError::CapacityExceeded(format!(
                "erase of {id} would remove {} memories (max_cascade {})",
                order.len(),
                opts.max_cascade
            )));
        }
        Ok(ErasePlan {
            order,
            nodes,
            ancestors_kept,
        })
    }

    /// Remove one memory everywhere [`Self::forget`] does, zeroing its
    /// record and its index vectors. Returns the HNSW nodes scrubbed. Same
    /// lock order as forget: each subsystem's lock is taken alone.
    fn erase_one(&self, node: &MemoryNode, erasing: &HashSet<MemoryId>) -> MenteResult<usize> {
        let id = node.id;
        let scrubbed = self.index.erase_memory(id, node, erasing);
        if self
            .scope_counts_ready
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            self.scope_counts
                .write()
                .apply(node, -1, &self.cognitive_config.hidden_count_tags);
        }
        if let Some(page_id) = self.page_of(id) {
            self.storage.erase_memory(page_id)?;
        }
        self.note_write(id);
        self.pending_access.lock().remove(&id);
        self.graph.remove_memory(id);
        self.page_map.write().remove(&id);
        Ok(scrubbed)
    }

    /// Drop cached entries built from the erased memories: by id where the
    /// cache records one, by text overlap where it does not.
    fn drop_cached_text(&self, nodes: &[&MemoryNode], opts: &EraseOptions) -> usize {
        let ids: HashSet<MemoryId> = nodes.iter().map(|n| n.id).collect();
        let min = opts.text_match_min_bytes;
        let from_erased = |text: &str| nodes.iter().any(|n| text_matches(text, &n.content, min));
        let mut dropped = self.speculative.write().remove_where(|e| {
            e.memory_ids.iter().any(|m| ids.contains(m)) || from_erased(&e.context_text)
        });
        dropped += self
            .pain
            .write()
            .remove_where(|s| ids.contains(&s.memory_id) || from_erased(&s.description));
        {
            let mut trajectory = self.trajectory.write();
            dropped += trajectory.remove_turns_where(|t| {
                from_erased(&t.topic_summary) || t.open_questions.iter().any(|q| from_erased(q))
            });
            dropped += trajectory.transitions.remove_topics(from_erased);
        }
        dropped += self
            .phantom
            .write()
            .remove_where(|p| from_erased(&p.gap_description) || from_erased(&p.source_reference));
        dropped
    }

    /// The scrub steps that follow the records: free space, WAL, snapshots,
    /// legacy copies, then the pending marker goes. Run by [`Self::erase`]
    /// and by open after a crash interrupted one.
    fn finish_erase(&self, finish: &PendingErase) -> MenteResult<FinishedErase> {
        let mut done = FinishedErase::default();
        if finish.scrub_free_space {
            done.free_space_bytes_zeroed = Some(self.storage.scrub_free_space()?);
        } else {
            self.storage.checkpoint_scrubbed()?;
        }
        // The graph edge log is zeroed before it is truncated.
        self.graph.save_scrubbed(&self.path.join("graph"))?;
        self.write_snapshots()?;
        if finish.remove_legacy_copies {
            done.legacy_files_removed = remove_legacy_copies(&self.path)?;
        }
        PendingErase::clear(&self.path)?;
        Ok(done)
    }

    /// Finish an erase a crash interrupted after its first record went.
    /// Open already scrubbed the orphaned index entries; this drops cache
    /// entries built from memories that no longer exist and runs the rest
    /// of the scrub.
    pub(crate) fn finish_pending_erase(&self) -> MenteResult<()> {
        let Some(finish) = PendingErase::read(&self.path) else {
            return Ok(());
        };
        warn!("finishing an erase that a crash interrupted");
        let dropped = {
            let pm = self.page_map.read();
            self.speculative
                .write()
                .remove_where(|e| e.memory_ids.iter().any(|m| !pm.contains_key(m)))
        };
        let orphans: Vec<MemoryId> = {
            let graph = self.graph.graph();
            let pm = self.page_map.read();
            graph
                .node_ids()
                .iter()
                .copied()
                .filter(|id| !pm.contains_key(id))
                .collect()
        };
        for id in &orphans {
            self.graph.remove_memory(*id);
        }
        self.save_cognitive_snapshots();
        self.finish_erase(&finish)?;
        info!(
            dropped,
            orphans = orphans.len(),
            "interrupted erase finished"
        );
        Ok(())
    }
}

/// What the rest of an erase must do, persisted while records are being
/// erased. Holds options only, never ids or content.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingErase {
    scrub_free_space: bool,
    remove_legacy_copies: bool,
}

const PENDING_FILE: &str = "erase.pending";

/// Whether `dir` holds an erase a crash interrupted.
pub(crate) fn pending(dir: &Path) -> bool {
    dir.join(PENDING_FILE).exists()
}

impl PendingErase {
    /// Write the marker atomically and durably (temp file, fsync, rename,
    /// directory fsync), before the first record is erased.
    fn write(&self, dir: &Path) -> MenteResult<()> {
        use std::io::Write;
        let bytes =
            serde_json::to_vec(self).map_err(|e| MenteError::Serialization(e.to_string()))?;
        let tmp = dir.join(format!("{PENDING_FILE}.tmp"));
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, dir.join(PENDING_FILE))?;
        sync_dir(dir);
        Ok(())
    }

    /// The marker, or None when there is none. An unreadable marker still
    /// means an erase was interrupted: finish it with every step.
    fn read(dir: &Path) -> Option<Self> {
        let bytes = std::fs::read(dir.join(PENDING_FILE)).ok()?;
        Some(serde_json::from_slice(&bytes).unwrap_or(Self {
            scrub_free_space: true,
            remove_legacy_copies: true,
        }))
    }

    fn clear(dir: &Path) -> MenteResult<()> {
        match std::fs::remove_file(dir.join(PENDING_FILE)) {
            Ok(()) => {
                sync_dir(dir);
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

/// What [`MenteDb::finish_erase`] did, for the receipt.
#[derive(Default)]
struct FinishedErase {
    free_space_bytes_zeroed: Option<u64>,
    legacy_files_removed: Vec<String>,
}

/// Fsync a directory so a rename or removal in it is durable (best effort).
fn sync_dir(dir: &Path) {
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
}

/// The memories an erase removes, in the order it removes them.
struct ErasePlan {
    order: Vec<(MemoryId, EraseReason)>,
    nodes: HashMap<MemoryId, MemoryNode>,
    ancestors_kept: Vec<MemoryId>,
}

/// Every node reachable from `start` through `next`, excluding `start`.
fn bfs(start: MemoryId, next: &dyn Fn(MemoryId) -> Vec<MemoryId>) -> Vec<MemoryId> {
    let mut seen = HashSet::from([start]);
    let mut out = Vec::new();
    let mut queue = VecDeque::from([start]);
    while let Some(m) = queue.pop_front() {
        for n in next(m) {
            if seen.insert(n) {
                out.push(n);
                queue.push_back(n);
            }
        }
    }
    out
}

/// Every regular file under `dir`, recursively, as (relative name, path).
fn walk(dir: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            match e.file_type() {
                Ok(t) if t.is_dir() => stack.push(p),
                Ok(t) if t.is_file() => {
                    let rel = p
                        .strip_prefix(dir)
                        .unwrap_or(&p)
                        .to_string_lossy()
                        .into_owned();
                    out.push((rel, p));
                }
                _ => {}
            }
        }
    }
    out.sort();
    out
}

/// Open handles on the files an erase may replace, so their old inodes can
/// be zeroed once replaced. The record file and WAL are scrubbed by storage
/// itself and the lock file holds nothing.
#[derive(Default)]
struct ReplacedFiles {
    #[cfg(unix)]
    open: Vec<(PathBuf, std::fs::File, u64)>,
}

impl ReplacedFiles {
    #[cfg(unix)]
    fn capture(dir: &Path) -> Self {
        use std::os::unix::fs::MetadataExt;
        let skip = [DATA_FILE, WAL_FILE, "LOCK"];
        let open = walk(dir)
            .into_iter()
            // Legacy copies are zeroed where they are deleted.
            .filter(|(rel, _)| !skip.contains(&rel.as_str()) && !rel.ends_with(LEGACY_KEPT_SUFFIX))
            .filter_map(|(_, p)| {
                let f = std::fs::OpenOptions::new().write(true).open(&p).ok()?;
                let ino = f.metadata().ok()?.ino();
                Some((p, f, ino))
            })
            .collect();
        Self { open }
    }

    #[cfg(not(unix))]
    fn capture(_dir: &Path) -> Self {
        Self::default()
    }

    /// Zero every captured file that a rename replaced and nothing else
    /// links to; returns how many.
    #[cfg(unix)]
    fn zero_replaced(self) -> usize {
        use std::os::unix::fs::MetadataExt;
        let mut zeroed = 0;
        for (path, file, ino) in self.open {
            let still_named = std::fs::metadata(&path).is_ok_and(|m| m.ino() == ino);
            let Ok(meta) = file.metadata() else {
                continue;
            };
            if still_named || meta.nlink() > 0 {
                continue;
            }
            if zero_file(&file, meta.len()).is_ok() {
                zeroed += 1;
            }
        }
        zeroed
    }

    #[cfg(not(unix))]
    fn zero_replaced(self) -> usize {
        0
    }
}

/// Overwrite `len` bytes of `file` from the start with zeros and fsync.
fn zero_file(file: &std::fs::File, len: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let mut f = file;
    f.seek(SeekFrom::Start(0))?;
    let chunk = vec![0u8; (len as usize).min(1 << 20)];
    let mut done = 0u64;
    while done < len {
        let n = ((len - done) as usize).min(chunk.len());
        f.write_all(&chunk[..n])?;
        done += n as u64;
    }
    f.sync_data()
}

/// Overwrite and delete the `.v1` files a format migration kept; returns
/// their names. A file with another hard link is only unlinked.
fn remove_legacy_copies(dir: &Path) -> MenteResult<Vec<String>> {
    let mut removed = Vec::new();
    for base in [LEGACY_PAGE_FILE, LEGACY_WAL_FILE, LEGACY_MAP_FILE] {
        let name = format!("{base}{LEGACY_KEPT_SUFFIX}");
        let p = dir.join(&name);
        let Ok(meta) = std::fs::metadata(&p) else {
            continue;
        };
        #[cfg(unix)]
        let shared = {
            use std::os::unix::fs::MetadataExt;
            meta.nlink() > 1
        };
        #[cfg(not(unix))]
        let shared = false;
        if !shared {
            let f = std::fs::OpenOptions::new().write(true).open(&p)?;
            zero_file(&f, meta.len())?;
        }
        std::fs::remove_file(&p)?;
        removed.push(name);
    }
    if !removed.is_empty() {
        sync_dir(dir);
        info!(?removed, "removed legacy copies");
    }
    Ok(removed)
}

/// One byte pattern the scan looks for.
struct Needle {
    memory: MemoryId,
    kind: &'static str,
    finder: memchr::memmem::Finder<'static>,
}

/// Scan every file under `dir` for the erased memories' content (as stored
/// raw and as JSON escapes it) and embedding bytes. WAL files are also read
/// entry by entry, since their payloads are compressed.
fn scan_directory(dir: &Path, nodes: &[&MemoryNode], opts: &EraseOptions) -> Verification {
    let mut v = Verification::default();
    let mut needles = Vec::new();
    for n in nodes {
        if n.content.len() >= opts.verify_min_content_bytes {
            let raw = n.content.as_bytes().to_vec();
            let escaped = serde_json::to_string(&n.content)
                .map(|s| s.as_bytes()[1..s.len() - 1].to_vec())
                .unwrap_or_default();
            if !escaped.is_empty() && escaped != raw {
                needles.push(Needle {
                    memory: n.id,
                    kind: "content",
                    finder: memchr::memmem::Finder::new(&escaped).into_owned(),
                });
            }
            needles.push(Needle {
                memory: n.id,
                kind: "content",
                finder: memchr::memmem::Finder::new(&raw).into_owned(),
            });
        } else {
            v.content_not_checked.push(n.id);
        }
        if !n.embedding.is_empty() {
            needles.push(Needle {
                memory: n.id,
                kind: "embedding",
                finder: memchr::memmem::Finder::new(&embedding_bytes(&n.embedding)).into_owned(),
            });
        }
    }
    let overlap = needles
        .iter()
        .map(|n| n.finder.needle().len())
        .max()
        .unwrap_or(0)
        .saturating_sub(1);
    let mut hits: HashSet<(String, MemoryId, &'static str)> = HashSet::new();
    for (rel, path) in walk(dir) {
        if rel == "LOCK" {
            continue;
        }
        match scan_file(&path, &needles, overlap, opts.verify_chunk_bytes) {
            Ok((bytes, found)) => {
                v.files_scanned += 1;
                v.bytes_scanned += bytes;
                for i in found {
                    hits.insert((rel.clone(), needles[i].memory, needles[i].kind));
                }
            }
            Err(_) => v.unreadable.push(rel.clone()),
        }
        let is_wal = rel.ends_with(".wal") || rel.starts_with(LEGACY_WAL_FILE);
        if is_wal && let Ok(entries) = mentedb_storage::Wal::read_entries(&path) {
            for e in entries {
                for (i, n) in needles.iter().enumerate() {
                    if n.finder.find(&e.data).is_some() {
                        hits.insert((rel.clone(), needles[i].memory, needles[i].kind));
                    }
                }
            }
        }
    }
    let mut hits: Vec<ResidualHit> = hits
        .into_iter()
        .map(|(file, memory, kind)| ResidualHit {
            file,
            memory,
            kind: kind.to_string(),
        })
        .collect();
    hits.sort_by(|a, b| (&a.file, a.memory, &a.kind).cmp(&(&b.file, b.memory, &b.kind)));
    v.residual_found = !hits.is_empty();
    v.hits = hits;
    v
}

/// Indexes of the needles found in the file, and the bytes read. Reads in
/// chunks that overlap by `overlap` bytes so a match across two reads is
/// still seen.
fn scan_file(
    path: &Path,
    needles: &[Needle],
    overlap: usize,
    chunk: usize,
) -> std::io::Result<(u64, HashSet<usize>)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut found = HashSet::new();
    let mut buf: Vec<u8> = Vec::with_capacity(chunk.max(1) + overlap);
    let mut read = vec![0u8; chunk.max(1)];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut read)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        buf.extend_from_slice(&read[..n]);
        for (i, needle) in needles.iter().enumerate() {
            if !found.contains(&i) && needle.finder.find(&buf).is_some() {
                found.insert(i);
            }
        }
        if buf.len() > overlap {
            buf.drain(..buf.len() - overlap);
        }
    }
    Ok((total, found))
}
