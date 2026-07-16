# Horizontal scale: design

MenteDB's engine is **single-writer per database directory**. Each tenant is a
directory (WAL, page store, HNSW index, in-memory page map). One process may
hold a tenant at a time; two writers on one directory interleave cached state
and corrupt it. This is by design, not a bug: making the storage engine
multi-writer would mean rebuilding the WAL, buffer pool, and index for
cross-process concurrency.

So "horizontal scale" does **not** mean many writers per tenant. It means many
*tenants* spread across many *tasks*, each tenant owned by exactly one task at a
time. Three things are required.

## 1. Safety: a cross-host single-writer lock (DONE, this repo)

The hosted deployment stores tenant directories on Amazon EFS (NFSv4) mounted by
several Fargate tasks. The old lock was `flock` (`fs2`), which is **host-local**
on NFS: a second host does not see it, so two tasks could open one tenant and
corrupt it.

`crates/mentedb-storage/src/lock.rs` now uses **`fcntl` open-file-description
locks (`F_OFD_SETLK`) on Linux**, which the NFSv4 lock manager enforces **across
hosts** on EFS, while keeping the per-open-file-description semantics `flock`
had. The lock releases on fd close or process death (a crashed task frees the
tenant after the NFS lease expires); a merely paused task keeps it, so there is
no zombie-write window. macOS/Windows keep `flock` (never on EFS).

This makes running >1 task **safe** (no corruption). It does **not** by itself
make it *functional*: a task that does not own a tenant simply fails to open it.
Which is why routing is required.

## 2. Functionality: tenant → task routing (platform, TODO)

Each request must reach the task that owns (or can acquire) that tenant.

- **Ownership map**: a DynamoDB table `tenant -> {task_id, address, lease_expiry,
  fence}`. A task writes its row when it acquires a tenant's engine lock and
  heartbeats `lease_expiry`; it deletes the row on release. Conditional writes
  make acquisition atomic.
- **Routing**: an incoming request for tenant `T` on task `A`:
  1. `A` reads the ownership map. If `A` owns `T` (or the row is absent/expired),
     `A` acquires the engine lock (authoritative) and serves it.
  2. If task `B` owns `T`, `A` forwards the request to `B`'s address (internal
     HTTP), or returns a redirect the ALB/consistent-hash layer follows.
- **Placement**: consistent-hash the tenant id onto the live task ring so most
  requests land on the owner directly and only rebalancing needs forwarding.

The engine lock is the source of truth; the DynamoDB map is a routing hint. If
they disagree (stale map), the engine lock still prevents double-open, so the
worst case is a failed open + retry, never corruption.

## 3. Elasticity: rebalancing on scale up/down (platform, TODO)

On scale-out/in, tenants move. Handoff must be **release-before-acquire**: the
losing task flushes, `close()`s (releasing the OFD lock and deleting its map
row), and only then may the gaining task acquire. The OFD lock guarantees the
gaining task cannot open until the losing one has released, so a botched handoff
degrades to unavailability, not corruption. Drain with the ALB dereg delay as
today.

## Validation gates (before enabling >1 task in prod)

1. **Cross-host lock test on real EFS**: two tasks on two hosts, same EFS mount;
   task B's open of a tenant task A holds must fail with "locked by another
   process". (Cannot be tested on a single host or in CI; OFD-over-NFS must be
   confirmed on EFS specifically.)
2. **Crash recovery**: kill task A holding tenant T; confirm B can acquire after
   the NFS lock lease expires, and measure that window (sets the failover SLO).
3. **Routing correctness** under a rebalance: no request for T is served by two
   tasks in the same window.

## Why this is staged, not shipped at once

Step 1 (the lock) is the corruption boundary and is safe to ship now: it is
behavior-preserving on one task and only *adds* cross-host enforcement. Steps 2
and 3 are a distributed-systems build where a routing/handoff bug is a
data-availability problem, not a corruption one (the lock protects correctness),
so they can be rolled out incrementally behind a flag once the EFS validation
above passes.
