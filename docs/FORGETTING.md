# Forgetting

MenteDB has two ways to delete a memory.

- **`forget`** removes a memory from recall: the indexes, the graph and the
  record table. It is fast and it is logged, so the memory never comes back
  after a crash. It does not follow links to memories derived from it, and
  it does not overwrite anything: the bytes stay on disk until the space is
  reused.
- **`erase`** removes the memory, everything derived from it, and every copy
  of their bytes the database holds, then returns a receipt you can check.
  Use it when a person asks for their data to be deleted.

This page covers `erase`: what it removes, what it cannot reach, and how to
verify it.

## Calling it

Rust:

```rust,no_run
use mentedb::{EraseOptions, MenteDb};
# fn run(db: &MenteDb, id: mentedb::prelude::MemoryId) -> mentedb::prelude::MenteResult<()> {
let receipt = db.erase(id, EraseOptions::default())?;
assert!(!receipt.verification.unwrap().residual_found);
# Ok(()) }
```

REST (needs a token with the write scope that owns the memory):

```bash
curl -X POST https://localhost:6677/v1/db/alice/memories/$ID/erase \
  -H "Authorization: Bearer $TOKEN" \
  -H "content-type: application/json" \
  -d '{"include_sources": false, "verify": true}'
```

The body is optional. It accepts any field of `EraseOptions` (below). The
response is the receipt. An unknown id is `404`, a cascade over
`max_cascade` is `409` with nothing changed. The same call is available on
`/v1/memories/{id}/erase` for the default database, as `MemoryService.Erase`
over gRPC, as `db.erase(id, include_sources=False, verify=True)` in the
Python SDK and as `db.erase(id, { includeSources, verify })` in the
TypeScript SDK.

## What is erased

### The cascade

Memories record what they were made from with `Derived` edges, from the
derived memory to its source. Erase follows them toward everything derived
from the memory, transitively:

| Derived memory | Made from | Erased with it |
|---|---|---|
| Extracted fact | The turn it was extracted from | yes |
| Consolidated memory | Each memory it merged | yes |
| Dedup keeper | The exact duplicate it absorbed | yes |
| Entity node | Each memory that mentions the entity | yes |
| Community summary | Each entity in the community | yes |

So erasing a turn erases the facts taken from it, the consolidations and
entity nodes built on those facts, and the community summaries built on
those entities. Entity and community nodes are shared: they also summarize
memories that are not erased. They are erased anyway, because their text
was written from the erased memory too. The next enrichment run rebuilds
them from the members that remain.

The memories the erased one was made from (its sources, such as the turn a
fact came from) are kept by default and listed in the receipt as
`ancestors_kept`, since they may still hold the content the erased memory
was made from. Set `include_sources` to erase them too; everything derived
from them goes with them, but other memories of a shared entity are never
pulled in. One exception applies either way: a linked memory with byte
identical content (an exact duplicate) is erased, with reason `identical`.

### Every copy of the bytes

| Where the bytes were | What erase does |
|---|---|
| Record file (`records.db`), the live record | The free and the overwrite with zeros are one WAL entry, fsynced before the file is touched; replay after a crash zeroes the extent again |
| Record file free space: older versions of the record (an update that outgrew its extent moved it) and records deleted earlier with `forget` | Every free extent is zeroed (`scrub_free_space`). The pass first checkpoints, then writes one free header per region and fsyncs, then zeroes, so a crash at any point leaves a file whose extent chain is intact |
| WAL (`records.wal`), the record's writes since the last checkpoint | Forced checkpoint; the old log file is overwritten with zeros through a handle still open on it, after the new log replaced it |
| Record map (`records_map.bin`) | Rewritten at the checkpoint; holds ids and offsets, never content |
| Vector index, the HNSW node | Every node that held the id (the current one and the old vectors an update left behind) gets its vector replaced with the mean of its neighbours' vectors, its id cleared and its lookup entry dropped. The node stays as a routing hop, so search quality is unchanged (see below) |
| Keyword index (BM25) | Postings and the document's term list removed; terms no other memory uses go |
| Tag index | The id is retired and every tag no other memory carries is dropped, since tags can carry names |
| Temporal and salience indexes | Entries removed |
| Index, graph and cache snapshots (`indexes/`, `graph/`, `mode_probes.bin`, `salience_bounds.bin`, `scope_counts.json`, `cognitive/`) | Rewritten without the erased memories; each replaced file's old version is overwritten with zeros once no name points at it (`zero_replaced_files`, Unix) |
| Graph edge log (`graph/edges.jsonl`) | Overwritten with zeros, then truncated, after the new snapshot is durable |
| Record cache, mode probes, salience bounds, pending access counts | Dropped |
| Speculative pre-assembly cache | Entries built from an erased memory, or whose text contains its content, dropped |
| Pain signals | Signals for an erased memory dropped |
| Trajectory turns, topic transitions, learned topic labels, phantom gaps | Entries whose text contains the erased content, or is contained in it, dropped. Text shorter than `text_match_min_bytes` is not matched |
| Legacy copies (`pages.db.v1`, `wal.log.v1`, `memory_map.bin.v1`, a full copy of the database a format migration kept) | Overwritten with zeros and deleted (`remove_legacy_copies`) |

### Places that hold no content

- `LOCK` is empty.
- `scope_counts.json` and the stats snapshot blob hold counts (and project
  names from tags).
- `enrichment_watermark` and the embedder calibration file hold numbers.
- The entity resolver (`cognitive/entities.json`) holds entity names and
  aliases learned across all memories, not memory text. A name is not
  removed because one memory that mentioned it was erased.
- The cognition stream buffers tokens a caller streams in for one turn; it
  is never written to disk and holds no stored memory.

## Search after an erase

A tombstoned HNSW node keeps routing searches, so its vector cannot simply
be dropped. Erase replaces it with the mean of the node's neighbours, which
sits where the node sat in the graph and is made only from vectors that
remain. Measured on 2,000 random 32 dimension vectors with a fifth of them
removed (`erase_keeps_recall_of_the_remaining_vectors` in
`crates/mentedb-index/src/hnsw.rs`), recall at 10 of the remaining vectors:

| Removal | Recall@10 |
|---|---|
| Plain tombstone (original vector kept) | 0.9385 |
| Erase (neighbour mean) | 0.9460 |
| Zeroed vector (the naive scrub) | 0.8865 |

`recall_of_unrelated_queries_is_unchanged_by_erase` in
`crates/mentedb/tests/erase.rs` checks that results for queries the erased
memories did not answer are identical before and after, and after a reopen.

## The receipt

```json
{
  "requested": "6f1c...",
  "erased": [
    {"id": "6f1c...", "reason": "requested", "content_sha256": "9b2e...", "embedding_sha256": "40aa..."},
    {"id": "a3d0...", "reason": "derived", "content_sha256": "c71f...", "embedding_sha256": "e5b2..."}
  ],
  "ancestors_kept": ["0b77..."],
  "profiles_to_review": ["5e19..."],
  "erased_at_us": 1793845123000000,
  "steps": {
    "records_zeroed": 2,
    "index_vectors_scrubbed": 3,
    "free_space_bytes_zeroed": 18944,
    "wal_scrubbed": true,
    "snapshots_rewritten": true,
    "replaced_files_zeroed": 9,
    "legacy_files_removed": [],
    "cache_entries_dropped": 1
  },
  "verification": {
    "files_scanned": 14,
    "bytes_scanned": 1203456,
    "residual_found": false,
    "hits": [],
    "unreadable": [],
    "content_not_checked": []
  }
}
```

It never holds content: only ids, SHA-256 hashes of each erased memory's
content and embedding (so you can show that a specific text was erased
without keeping it), what each step did, and the scan result.

`profiles_to_review` lists the user profile memories of the erased memories'
owners. A profile is a summary written across many memories with no link to
each, so it may still paraphrase what was erased. Review it, or rebuild it
with the next enrichment run.

## Verifying

With `verify` (on by default) erase scans every file in the database
directory after the scrub for:

- each erased memory's content, as stored and as JSON escapes it;
- each erased memory's embedding as raw little endian f32 bytes.

WAL files are also read entry by entry, since their payloads are compressed.
Any match is reported in `hits` with the file and the memory, and
`residual_found` is true. The scan reads the files themselves; it does not
trust the steps. Erasing with `remove_legacy_copies` off and a `.v1` file
present, or with `scrub_free_space` off after the record had moved, shows
the copies left behind (`verification_finds_what_a_skipped_step_leaves`).

Content shorter than `verify_min_content_bytes` (16 by default) is not
searched for, since it would match unrelated data; those ids are listed in
`content_not_checked`, and their embedding is still searched. A hit can
also be another memory that quotes the erased text: the scan reports bytes,
not memories.

To check by hand after closing the database:

```bash
grep -rl --binary-files=text 'the exact content' /path/to/db || echo clean
```

## Crash safety

- A crash before the first record's WAL entry is fsynced leaves everything
  as it was. Erase the memory again.
- Derived memories are erased first and the requested one last. A crash
  part way through the cascade leaves the requested memory in place, so
  erasing it again finds and finishes the rest.
- Before the first record goes, erase writes an `erase.pending` marker
  (options only, no ids or content). If the process dies after that, the
  next open scrubs the orphaned index entries, drops cache entries for
  memories that no longer exist, removes orphaned graph nodes, zeroes free
  space, checkpoints and zeroes the old WAL, rewrites the snapshots, removes
  the legacy copies, and deletes the marker.
- An erased memory never comes back: its free is in the WAL before the file
  changes, and replay redoes the zeroing.
- A neighbour is never touched: every write lands inside the erased extent
  or inside free space, and the free space pass writes the headers that
  keep the extent chain walkable before it zeroes.

## Options

| Field | Default | Meaning |
|---|---|---|
| `include_sources` | `false` | Also erase what the memory was derived from |
| `verify` | `true` | Scan the directory afterwards |
| `scrub_free_space` | `true` | Zero all record file free space, not only the erased extents. The storage lock is held while it runs, so writes wait; it takes time proportional to the free space |
| `remove_legacy_copies` | `true` | Overwrite and delete the `.v1` files |
| `zero_replaced_files` | `true` | Zero the old version of each replaced snapshot file (Unix; files with another hard link are left alone) |
| `max_cascade` | `10000` | Refuse, before changing anything, an erase that would remove more memories |
| `text_match_min_bytes` | `16` | Shortest cached text matched against erased content |
| `verify_min_content_bytes` | `16` | Shortest content the scan searches for |
| `verify_chunk_bytes` | `1048576` | Read size of the scan |

## Cost

An erase does whole database work on top of removing its memories: it
rewrites the index, graph and cognitive snapshots, zeroes all record file
free space (with `scrub_free_space`), and reads every file in the directory
(with `verify`). The time and the bytes written and read grow with the size
of the database, not the size of the cascade. That suits erase requests one
at a time. To remove a whole tenant, delete its database directory instead of
erasing memories one by one.

## What erase cannot reach

Erase covers the files in the database directory as the operating system
shows them. It cannot reach:

- **Backups and copies outside the directory**: `.mentebackup` files,
  volume snapshots, replicas, copies a host made. Erase in each, or expire
  them.
- **Hard links**: a file in the directory hard linked from elsewhere is
  left as it is and stays readable from the other name.
- **Storage below the file system**: copy on write file systems (APFS, ZFS,
  Btrfs) and their snapshots, journals, SSD wear leveling and network
  storage may keep old blocks after a file is overwritten. Use encrypted
  storage and destroy the key when that matters.
- **Process memory**: freed memory is not wiped, and swap may hold pages.
- **Paraphrases**: a memory that restates the content in other words with
  no `Derived` link (the user profile is the common case, listed in
  `profiles_to_review`). The scan finds exact bytes only.
- **Writes during the erase**: a memory derived from an erased one while the
  erase runs (an enrichment pass in flight) is not part of its cascade.
  Erase when no enrichment is running for the account, or erase again.
- **Host data**: blobs a host stores with `write_blob`, embedding caches
  inside an embedder, and anything an application logged. The scan does
  report blobs in the directory that still hold the bytes.
