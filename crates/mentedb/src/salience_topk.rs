//! Top k for `ORDER BY salience DESC LIMIT k` MQL scans without loading every
//! candidate.
//!
//! `salience` in MQL is the decayed value at the query's instant, so the plain
//! path loads every candidate memory to compute it. That value is a function
//! of a handful of stored fields: the base salience, `created_at`,
//! `accessed_at`, `access_count`, and whether the memory is decay exempt
//! (a tag or attribute). This module caches exactly those inputs per memory,
//! computes each candidate's key with the same function the full path uses
//! ([`MenteDb::decayed_salience_of`]), and loads candidates from the highest
//! key down, stopping once the next key is below the current k-th result.
//!
//! # Why the stored base alone is not a bound
//!
//! For a decaying memory the value is `clamp(s * f + b, min, max)` where
//! `f = 2^(-t / half_life)` is in `[0, 1]` and `b = access_boost * ln(1 + n)`.
//! The access bonus `b` and the `min_salience` floor both lift the value above
//! the stored base `s` (a base of 0.05 with 50 accesses is worth about 0.44;
//! a base of 0.001 is worth the 0.01 floor). Ordering by `s` would therefore
//! stop too early and drop real results. With every input cached, the key is
//! the decayed value itself, so `key >= effective` holds with equality and the
//! stop rule is exact.
//!
//! # Cache freshness
//!
//! A cached entry is dropped by `note_write` whenever its memory is written
//! (every storage write path calls it), with the same pending token protocol
//! as the mode probes, so a copy loaded before a concurrent write is never
//! cached. A memory with no entry has an unknown key and is always loaded.
//! The cache is persisted with the snapshots, each entry stamped with the page
//! and write stamp it was loaded at, and restored on open only where nothing
//! wrote the memory since and the decay exemption settings are unchanged.
//!
//! # Warming
//!
//! A query fills the cache only for the memories it loads, so the first
//! `ORDER BY salience DESC LIMIT k` on a cold account loads every memory on
//! the request path. [`MenteDb::warm_salience_bounds`] fills it off the
//! request path instead, in chunks, with the same pending token protocol, so
//! a write landing mid warm still wins and a query never waits for it.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};

use super::*;

/// The stored fields a memory's decayed salience is computed from.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundEntry {
    salience: f32,
    created_at: u64,
    accessed_at: u64,
    access_count: u32,
    /// `is_decay_exempt` under the settings in force when it was loaded.
    exempt: bool,
    /// Page and storage write stamp read BEFORE the memory was loaded, or
    /// None when storage had none (the entry is then never persisted).
    stamp: Option<(PageId, u64)>,
}

/// A decay input cache slot: being loaded (by the holder of this token) or
/// ready.
pub(crate) enum BoundSlot {
    Loading(u64),
    Ready(BoundEntry),
}

static NEXT_BOUND_TOKEN: AtomicU64 = AtomicU64::new(1);

/// Settings for [`MenteDb::warm_salience_bounds`].
#[derive(Debug, Clone)]
pub struct SalienceWarmConfig {
    /// Memories loaded per chunk. Locks are held only to claim a chunk and to
    /// publish its entries, never across a page load, and cancellation is
    /// checked between chunks, so this bounds how long a cancelled warm keeps
    /// reading. Default: 256. Zero is treated as one.
    pub chunk_size: usize,
    /// Sleep between chunks, leaving storage bandwidth to foreground work.
    /// Zero only yields the thread. Default: zero.
    pub pause: Duration,
    /// Write `salience_bounds.bin` once the warm completes, so a crash before
    /// the next snapshot keeps the work. Skipped when cancelled or when
    /// nothing new was cached. Default: true.
    pub persist: bool,
}

impl Default for SalienceWarmConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256,
            pause: Duration::ZERO,
            persist: true,
        }
    }
}

/// What a [`MenteDb::warm_salience_bounds`] call did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SalienceWarmReport {
    /// Memories in the database when the warm started.
    pub memories: usize,
    /// Of those, memories that already had an entry or a load in flight,
    /// including ones a concurrent query or warm cached before this call
    /// reached them.
    pub already_cached: usize,
    /// Memories this call loaded from storage.
    pub loaded: usize,
    /// Entries this call added to the cache.
    pub cached: usize,
    /// Loads discarded because the memory was written (or deleted) while
    /// loading: the write wins and the memory stays uncached.
    pub invalidated: usize,
    /// Loads that failed or found a different memory on the page.
    pub unreadable: usize,
    /// Whether `should_continue` stopped the warm before it finished.
    pub cancelled: bool,
    /// Whether this call wrote `salience_bounds.bin`.
    pub persisted: bool,
    /// Wall time of the call.
    pub elapsed: Duration,
}

/// A memory claimed for loading by a warm chunk: its page and the pending
/// token its slot holds.
pub(crate) struct WarmClaim {
    id: MemoryId,
    page: PageId,
    token: u64,
}

/// A claimed memory after its load: the entry to publish, None when the load
/// failed.
pub(crate) struct WarmLoaded {
    id: MemoryId,
    token: u64,
    entry: Option<BoundEntry>,
}

/// File (in the database directory) holding the persisted decay inputs.
pub(crate) const SALIENCE_BOUNDS_FILE: &str = "salience_bounds.bin";
const BOUNDS_MAGIC: &[u8; 8] = b"MDBSALB1";
/// Bytes per persisted record: id, page, stamp, salience, created, accessed,
/// access count, exempt flag.
const BOUND_RECORD_LEN: usize = 16 + 8 + 8 + 4 + 8 + 8 + 4 + 1;

/// A candidate with a cached key, ordered for a max heap: highest key first,
/// then lowest page (the order a tie is broken in the results).
struct Visit {
    key: f32,
    page: PageId,
}

impl PartialEq for Visit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Visit {}

impl PartialOrd for Visit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Visit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .total_cmp(&other.key)
            .then_with(|| other.page.0.cmp(&self.page.0))
    }
}

/// The best k matches seen so far, ordered by decayed salience descending,
/// ties by page ascending: the order the full path's stable sort over a
/// page ordered scan produces.
struct TopK {
    k: usize,
    items: Vec<(f32, PageId, ScoredMemory)>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::new(),
        }
    }

    /// The k-th best decayed salience, once k results are held.
    fn floor(&self) -> Option<f32> {
        if self.items.len() >= self.k {
            self.items.last().map(|(e, _, _)| *e)
        } else {
            None
        }
    }

    fn offer(&mut self, eff: f32, page: PageId, memory: MemoryNode) {
        let pos = self
            .items
            .partition_point(|(e, p, _)| *e > eff || (*e == eff && p.0 < page.0));
        if pos >= self.k {
            return;
        }
        self.items
            .insert(pos, (eff, page, ScoredMemory { memory, score: 1.0 }));
        self.items.truncate(self.k);
    }

    fn into_results(self) -> Vec<ScoredMemory> {
        self.items.into_iter().map(|(_, _, sm)| sm).collect()
    }
}

/// Fingerprint of the decay exemption settings, so persisted exempt flags
/// are trusted only under the settings that produced them.
fn exemption_fingerprint(cfg: &CognitiveConfig) -> u32 {
    let mut h = crc32fast::Hasher::new();
    for t in &cfg.decay_exempt_tags {
        h.update(&(t.len() as u64).to_le_bytes());
        h.update(t.as_bytes());
    }
    match &cfg.decay_exempt_attribute {
        Some(a) => {
            h.update(&[1]);
            h.update(a.as_bytes());
        }
        None => h.update(&[0]),
    }
    h.finalize()
}

impl MenteDb {
    /// The decayed salience at `now` of the memory `e` describes, computed as
    /// the full path computes it. NaN reads as infinity, so such a memory is
    /// always loaded (and then sends the query to the full path).
    pub(crate) fn bound_key(&self, e: &BoundEntry, now: u64) -> f32 {
        let v = self.decayed_salience_of(
            e.exempt,
            e.salience,
            e.created_at,
            e.accessed_at,
            e.access_count,
            now,
        );
        if v.is_nan() { f32::INFINITY } else { v }
    }

    /// Whether `ORDER BY ob` can take the top k walk.
    pub(crate) fn salience_top_k_applies(&self, ob: &OrderBy) -> bool {
        self.cognitive_config.mql_salience_top_k && ob.field == Field::Salience && ob.descending
    }

    /// The k memories among `candidates` (every memory when None) that
    /// satisfy the WHERE clause, ordered by decayed salience at `now`
    /// descending, ties by page ascending: exactly what loading every
    /// candidate, filtering, stable sorting by decayed salience and
    /// truncating to k returns.
    ///
    /// Candidates with no cached decay inputs are loaded first, then the rest
    /// in descending key order until the next key falls below the k-th
    /// result. Returns None (run the full path) when a matching memory's
    /// decayed salience is NaN, which the full path's sort leaves unordered.
    pub(crate) fn salience_top_k(
        &self,
        candidates: Option<&HashSet<MemoryId>>,
        filters: &[Filter],
        condition: Option<&Condition>,
        k: usize,
        now: u64,
    ) -> Option<Vec<ScoredMemory>> {
        if k == 0 {
            return Some(Vec::new());
        }
        // Page ids copied out under a brief page map read.
        let pages: Vec<(MemoryId, PageId)> = match candidates {
            Some(ids) => self.pages_of(ids),
            None => {
                let pm = self.page_map.read();
                pm.iter().map(|(id, p)| (*id, *p)).collect()
            }
        };
        // Split by whether the inputs are cached, under a brief cache read.
        let mut unknown: Vec<(MemoryId, PageId)> = Vec::new();
        let mut known: Vec<Visit> = Vec::new();
        {
            let cache = self.salience_bounds.read();
            for (id, page) in pages {
                match cache.get(&id) {
                    Some(BoundSlot::Ready(e)) => known.push(Visit {
                        key: self.bound_key(e, now),
                        page,
                    }),
                    _ => unknown.push((id, page)),
                }
            }
        }

        let mut top = TopK::new(k);
        unknown.sort_unstable_by_key(|(_, p)| p.0);
        for (id, page) in unknown {
            let Some(memory) = self.load_caching_bound(id, page) else {
                continue;
            };
            if !self.consider(&mut top, memory, page, filters, condition, now) {
                return None;
            }
        }

        let mut heap = BinaryHeap::from(known);
        while let Some(next) = heap.pop() {
            // Every unvisited candidate's decayed salience is at most this
            // key; one strictly below the k-th result can never displace it
            // (an equal one could, on a lower page, so it is still visited).
            if top.floor().is_some_and(|floor| next.key < floor) {
                break;
            }
            let Ok(memory) = self.storage.load_memory(next.page) else {
                continue;
            };
            if !self.consider(&mut top, memory, next.page, filters, condition, now) {
                return None;
            }
        }
        Some(top.into_results())
    }

    /// Offer a loaded memory to `top` if it matches the WHERE clause. False
    /// when it matches with a NaN decayed salience (the caller falls back).
    fn consider(
        &self,
        top: &mut TopK,
        memory: MemoryNode,
        page: PageId,
        filters: &[Filter],
        condition: Option<&Condition>,
        now: u64,
    ) -> bool {
        let keep = match condition {
            Some(cond) => self.condition_matches(cond, &memory, now),
            None => filters.iter().all(|f| self.filter_matches(f, &memory, now)),
        };
        if !keep {
            return true;
        }
        let eff = self.decayed_salience_at(&memory, now);
        if eff.is_nan() {
            return false;
        }
        top.offer(eff, page, memory);
        true
    }

    /// Load the memory on `page` and cache its decay inputs. The slot is
    /// marked pending while loading; a write to this memory removes the slot,
    /// so a copy read before that write is returned but never cached.
    fn load_caching_bound(&self, id: MemoryId, page: PageId) -> Option<MemoryNode> {
        let token = {
            let mut cache = self.salience_bounds.write();
            match cache.get(&id) {
                Some(_) => None,
                None => {
                    let token = NEXT_BOUND_TOKEN.fetch_add(1, AtomicOrdering::Relaxed);
                    cache.insert(id, BoundSlot::Loading(token));
                    Some(token)
                }
            }
        };
        // Stamp first, then load: see `BoundEntry::stamp`.
        let stamp = self.bound_stamp(id, page);
        let loaded = self.storage.load_memory(page).ok();
        if let Some(token) = token {
            let entry = loaded.as_ref().and_then(|n| self.bound_entry(id, n, stamp));
            let mut cache = self.salience_bounds.write();
            if matches!(cache.get(&id), Some(BoundSlot::Loading(t)) if *t == token) {
                match entry {
                    Some(e) => {
                        cache.insert(id, BoundSlot::Ready(e));
                    }
                    None => {
                        cache.remove(&id);
                    }
                }
            }
        }
        loaded
    }

    /// The storage write stamp of `id` on `page`, read BEFORE the memory is
    /// loaded: see `BoundEntry::stamp`.
    fn bound_stamp(&self, id: MemoryId, page: PageId) -> Option<(PageId, u64)> {
        self.storage
            .memory_stamp(page)
            .filter(|(sid, _)| *sid == id)
            .map(|(_, s)| (page, s))
    }

    /// The decay inputs of `node`, None when the page held another memory.
    fn bound_entry(
        &self,
        id: MemoryId,
        node: &MemoryNode,
        stamp: Option<(PageId, u64)>,
    ) -> Option<BoundEntry> {
        (node.id == id).then(|| BoundEntry {
            salience: node.salience,
            created_at: node.created_at,
            accessed_at: node.accessed_at,
            access_count: node.access_count,
            exempt: self.is_decay_exempt(node),
            stamp,
        })
    }

    /// Fill the salience decay input cache for every memory that has no
    /// entry, so the next `ORDER BY salience DESC LIMIT k` query loads only
    /// its top candidates instead of every memory.
    ///
    /// Meant to run after open, on a background thread: it blocks for as long
    /// as loading the uncached memories takes (seconds per ten thousand on
    /// network storage), and returns at once when the cache is already
    /// complete (a reopen that restored `salience_bounds.bin`).
    ///
    /// Works in chunks of `config.chunk_size`: page ids are copied out under a
    /// brief page map read, the chunk's slots are claimed with pending tokens
    /// under a brief cache write, the pages are loaded with no lock held, and
    /// each entry is published only if its token is still in place. A write
    /// to a memory removes its slot (`note_write`), so a copy loaded before
    /// that write is discarded, never cached. Stores, recall, MQL and
    /// maintenance run alongside it, a query never waits for it (a memory
    /// being warmed reads as uncached and the query loads it itself), and
    /// concurrent or repeated calls are safe: a memory already claimed or
    /// cached is skipped.
    ///
    /// `should_continue` is called before each chunk; once it returns false
    /// the warm returns without another storage read and without persisting
    /// (the report says `cancelled`). Call it with a check that turns false
    /// before the database is closed or handed to another owner, and wait
    /// for the call to return before releasing the files. Apart from that
    /// final persist (`config.persist`, written atomically like the
    /// snapshots), the warm writes nothing to disk itself.
    pub fn warm_salience_bounds(
        &self,
        config: &SalienceWarmConfig,
        should_continue: &dyn Fn() -> bool,
    ) -> MenteResult<SalienceWarmReport> {
        let started = Instant::now();
        let mut report = SalienceWarmReport::default();
        let ids: Vec<MemoryId> = self.page_map.read().keys().copied().collect();
        report.memories = ids.len();
        let missing: Vec<MemoryId> = {
            let cache = self.salience_bounds.read();
            ids.into_iter()
                .filter(|id| !cache.contains_key(id))
                .collect()
        };
        report.already_cached = report.memories - missing.len();

        let mut chunks = missing.chunks(config.chunk_size.max(1)).peekable();
        while let Some(chunk) = chunks.next() {
            if !should_continue() {
                report.cancelled = true;
                break;
            }
            let claimed = self.warm_claim(chunk, &mut report);
            let loaded = self.warm_load(&claimed);
            report.loaded += loaded.len();
            self.warm_publish(loaded, &mut report);
            if chunks.peek().is_some() {
                if config.pause.is_zero() {
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(config.pause);
                }
            }
        }

        if config.persist && !report.cancelled && report.cached > 0 {
            if should_continue() {
                self.save_salience_bounds()?;
                report.persisted = true;
            } else {
                report.cancelled = true;
            }
        }
        report.elapsed = started.elapsed();
        Ok(report)
    }

    /// Claim the memories of `ids` that still have no slot: resolve their
    /// pages under a brief page map read, then mark each slot pending with a
    /// fresh token under a brief cache write. Memories that gained a slot
    /// since the warm started count as already cached.
    pub(crate) fn warm_claim(
        &self,
        ids: &[MemoryId],
        report: &mut SalienceWarmReport,
    ) -> Vec<WarmClaim> {
        let pages = self.pages_of(ids);
        let mut cache = self.salience_bounds.write();
        pages
            .into_iter()
            .filter_map(|(id, page)| {
                if cache.contains_key(&id) {
                    report.already_cached += 1;
                    return None;
                }
                let token = NEXT_BOUND_TOKEN.fetch_add(1, AtomicOrdering::Relaxed);
                cache.insert(id, BoundSlot::Loading(token));
                Some(WarmClaim { id, page, token })
            })
            .collect()
    }

    /// Load each claimed memory with no lock held, stamp read first.
    pub(crate) fn warm_load(&self, claimed: &[WarmClaim]) -> Vec<WarmLoaded> {
        claimed
            .iter()
            .map(|c| {
                let stamp = self.bound_stamp(c.id, c.page);
                let entry = self
                    .storage
                    .load_memory(c.page)
                    .ok()
                    .and_then(|n| self.bound_entry(c.id, &n, stamp));
                WarmLoaded {
                    id: c.id,
                    token: c.token,
                    entry,
                }
            })
            .collect()
    }

    /// Publish loaded entries under one brief cache write. An entry whose
    /// slot no longer holds its token was overtaken by a write and is
    /// dropped; a failed load clears its own pending slot.
    pub(crate) fn warm_publish(&self, loaded: Vec<WarmLoaded>, report: &mut SalienceWarmReport) {
        let mut cache = self.salience_bounds.write();
        for l in loaded {
            if !matches!(cache.get(&l.id), Some(BoundSlot::Loading(t)) if *t == l.token) {
                report.invalidated += 1;
                continue;
            }
            match l.entry {
                Some(e) => {
                    cache.insert(l.id, BoundSlot::Ready(e));
                    report.cached += 1;
                }
                None => {
                    cache.remove(&l.id);
                    report.unreadable += 1;
                }
            }
        }
    }

    /// Persist the ready decay inputs that carry a write stamp. Atomic (temp
    /// file, fsync, rename), and serialized, since snapshots, quick close and
    /// a finished warm share the temp file.
    pub(crate) fn save_salience_bounds(&self) -> MenteResult<()> {
        let _save = self.salience_bounds_save.lock();
        let bytes = {
            let cache = self.salience_bounds.read();
            encode_bounds(
                exemption_fingerprint(&self.cognitive_config),
                cache.iter().filter_map(|(id, slot)| match slot {
                    BoundSlot::Ready(e) => e.stamp.map(|s| (*id, *e, s)),
                    BoundSlot::Loading(_) => None,
                }),
            )
        };
        let path = self.path.join(SALIENCE_BOUNDS_FILE);
        let tmp = self.path.join(format!("{SALIENCE_BOUNDS_FILE}.tmp"));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// The persisted decay inputs still current for `mapped` (the memory map
    /// just loaded on open): an entry is restored only when its memory is on
    /// the same page with the same write stamp, i.e. nothing wrote it since,
    /// and the whole file only when the exemption settings are unchanged.
    pub(crate) fn load_salience_bounds(
        dir: &Path,
        mapped: &[mentedb_storage::MappedMemory],
        cfg: &CognitiveConfig,
    ) -> HashMap<MemoryId, BoundSlot> {
        let mut out = HashMap::new();
        let Ok(bytes) = std::fs::read(dir.join(SALIENCE_BOUNDS_FILE)) else {
            return out;
        };
        let Some((fingerprint, entries)) = decode_bounds(&bytes) else {
            warn!("salience bound snapshot unreadable, bounds will load on use");
            return out;
        };
        if fingerprint != exemption_fingerprint(cfg) {
            return out;
        }
        let current: HashMap<MemoryId, (PageId, u64)> =
            mapped.iter().map(|m| (m.id, (m.page, m.stamp))).collect();
        for (id, entry) in entries {
            if entry.stamp.is_some() && current.get(&id).copied() == entry.stamp {
                out.insert(id, BoundSlot::Ready(entry));
            }
        }
        out
    }
}

/// Encode stamped entries: magic, exemption fingerprint, count, records,
/// CRC32 of everything before.
fn encode_bounds(
    fingerprint: u32,
    entries: impl Iterator<Item = (MemoryId, BoundEntry, (PageId, u64))>,
) -> Vec<u8> {
    let mut records = Vec::new();
    let mut n = 0u64;
    for (id, e, (page, stamp)) in entries {
        records.extend_from_slice(id.0.as_bytes());
        records.extend_from_slice(&page.0.to_le_bytes());
        records.extend_from_slice(&stamp.to_le_bytes());
        records.extend_from_slice(&e.salience.to_le_bytes());
        records.extend_from_slice(&e.created_at.to_le_bytes());
        records.extend_from_slice(&e.accessed_at.to_le_bytes());
        records.extend_from_slice(&e.access_count.to_le_bytes());
        records.push(u8::from(e.exempt));
        n += 1;
    }
    let mut buf = Vec::with_capacity(BOUNDS_MAGIC.len() + 12 + records.len() + 4);
    buf.extend_from_slice(BOUNDS_MAGIC);
    buf.extend_from_slice(&fingerprint.to_le_bytes());
    buf.extend_from_slice(&n.to_le_bytes());
    buf.extend_from_slice(&records);
    let crc = crc32fast::hash(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

fn le_u64(b: &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(b.try_into().ok()?))
}

fn le_u32(b: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(b.try_into().ok()?))
}

/// Decode a bounds file into its fingerprint and entries; None when it is
/// malformed in any way.
fn decode_bounds(bytes: &[u8]) -> Option<(u32, Vec<(MemoryId, BoundEntry)>)> {
    let (body, crc) = bytes.split_at(bytes.len().checked_sub(4)?);
    if crc32fast::hash(body) != le_u32(crc)? {
        return None;
    }
    let rest = body.strip_prefix(BOUNDS_MAGIC.as_slice())?;
    let (fp, rest) = rest.split_at_checked(4)?;
    let (count, records) = rest.split_at_checked(8)?;
    let n = usize::try_from(le_u64(count)?).ok()?;
    if records.len() != n.checked_mul(BOUND_RECORD_LEN)? {
        return None;
    }
    let mut out = Vec::with_capacity(n);
    for r in records.as_chunks::<BOUND_RECORD_LEN>().0 {
        let id = MemoryId(uuid::Uuid::from_slice(r.get(0..16)?).ok()?);
        let page = PageId(le_u64(r.get(16..24)?)?);
        let stamp = le_u64(r.get(24..32)?)?;
        let salience = f32::from_bits(le_u32(r.get(32..36)?)?);
        let created_at = le_u64(r.get(36..44)?)?;
        let accessed_at = le_u64(r.get(44..52)?)?;
        let access_count = le_u32(r.get(52..56)?)?;
        let exempt = *r.get(56)? == 1;
        out.push((
            id,
            BoundEntry {
                salience,
                created_at,
                accessed_at,
                access_count,
                exempt,
                stamp: Some((page, stamp)),
            },
        ));
    }
    Some((le_u32(fp)?, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_file_roundtrips_and_rejects_corruption() {
        let id = MemoryId::new();
        let e = BoundEntry {
            salience: 0.37,
            created_at: 11,
            accessed_at: 22,
            access_count: 33,
            exempt: true,
            stamp: Some((PageId(7), 99)),
        };
        let bytes = encode_bounds(0xabcd, std::iter::once((id, e, (PageId(7), 99))));
        let (fp, got) = decode_bounds(&bytes).unwrap();
        assert_eq!(fp, 0xabcd);
        assert_eq!(got.len(), 1);
        let (gid, g) = got[0];
        assert_eq!(gid, id);
        assert_eq!(g.salience, 0.37);
        assert_eq!(
            (g.created_at, g.accessed_at, g.access_count, g.exempt),
            (11, 22, 33, true)
        );
        assert_eq!(g.stamp, Some((PageId(7), 99)));

        let mut bad = bytes.clone();
        bad[20] ^= 1;
        assert!(decode_bounds(&bad).is_none());
        assert!(decode_bounds(&bytes[..bytes.len() - 1]).is_none());
        assert!(decode_bounds(&[]).is_none());
    }
}
