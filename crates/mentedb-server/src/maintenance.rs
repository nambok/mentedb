//! Built-in maintenance sweep for self-hosted deployments.
//!
//! Runs the rule-based "overnight" jobs that keep memory healthy: consolidate
//! near-duplicates, decay salience, forget cold memories, and evict the stale
//! speculative cache. A self-hoster gets the same upkeep the cloud worker
//! provides, either automatically on a timer (`--maintenance-interval-hours`)
//! or one-shot via the `maintenance` subcommand for external cron, without
//! writing any code. Per-turn fact extraction is handled separately by
//! `--auto-extract`.

use std::sync::Arc;
use std::time::Duration;

use mentedb::MenteDb;
use mentedb::consolidation::archival::ArchivalDecision;
use tracing::{info, warn};

/// Minimum cluster size for consolidation.
const MIN_CLUSTER_SIZE: usize = 2;
/// Cosine similarity threshold for grouping memories to consolidate.
const SIMILARITY_THRESHOLD: f32 = 0.85;
/// Speculative cache entries older than this (24h) are evicted.
const CACHE_MAX_AGE_US: u64 = 24 * 3600 * 1_000_000;
/// Cap on deletions per sweep, so a single run can never mass-delete.
const MAX_FORGETS_PER_SWEEP: usize = 500;

/// What one maintenance sweep did.
#[derive(Debug, Default)]
pub struct MaintenanceReport {
    pub consolidated: usize,
    pub decayed: usize,
    pub forgotten: usize,
}

/// Run one maintenance sweep synchronously. Each job is best-effort: a failure
/// in one is logged and the remaining jobs still run.
pub fn run_sweep(db: &MenteDb) -> MaintenanceReport {
    let mut report = MaintenanceReport::default();

    // 1. Consolidation: merge near-duplicate memories into one richer memory
    //    (sources invalidated and linked, not deleted).
    match db.find_consolidation_candidates(MIN_CLUSTER_SIZE, SIMILARITY_THRESHOLD) {
        Ok(candidates) => {
            for c in candidates {
                if db.consolidate_cluster(&c.memories).is_ok() {
                    report.consolidated += 1;
                }
            }
        }
        Err(e) => warn!("maintenance: consolidation failed: {e}"),
    }

    // 2. Decay: age salience so old, unused memories rank lower over time.
    match db.apply_decay_global() {
        Ok(n) => report.decayed = n,
        Err(e) => warn!("maintenance: decay failed: {e}"),
    }

    // 3. Archival: forget cold memories flagged for deletion (capped per sweep).
    match db.evaluate_archival_global() {
        Ok(decisions) => {
            for (id, decision) in decisions {
                if report.forgotten >= MAX_FORGETS_PER_SWEEP {
                    break;
                }
                if decision == ArchivalDecision::Delete && db.forget(id).is_ok() {
                    report.forgotten += 1;
                }
            }
        }
        Err(e) => warn!("maintenance: archival failed: {e}"),
    }

    // 4. Evict stale speculative pre-assembly cache entries.
    db.evict_stale_speculative(CACHE_MAX_AGE_US);

    // 5. Persist everything to disk.
    if let Err(e) = db.flush() {
        warn!("maintenance: flush failed: {e}");
    }

    report
}

/// Spawn a background task that runs a sweep every `interval_hours`. A value of
/// 0 disables the scheduler (run the `maintenance` subcommand from cron instead).
pub fn spawn_scheduler(db: Arc<MenteDb>, interval_hours: u64) {
    if interval_hours == 0 {
        info!("maintenance scheduler disabled (--maintenance-interval-hours 0)");
        return;
    }
    info!("maintenance scheduler enabled: sweep every {interval_hours}h");
    tokio::spawn(async move {
        let interval = Duration::from_secs(interval_hours * 3600);
        loop {
            tokio::time::sleep(interval).await;
            let db = Arc::clone(&db);
            match tokio::task::spawn_blocking(move || run_sweep(&db)).await {
                Ok(r) => info!(
                    consolidated = r.consolidated,
                    decayed = r.decayed,
                    forgotten = r.forgotten,
                    "maintenance sweep complete"
                ),
                Err(e) => warn!("maintenance sweep task panicked: {e}"),
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb::prelude::*;
    use mentedb_embedding::hash_provider::HashEmbeddingProvider;

    #[test]
    fn sweep_runs_on_a_populated_db_without_error() {
        let path =
            std::env::temp_dir().join(format!("mentedb_maint_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let db = MenteDb::open_with_embedder(&path, Box::new(HashEmbeddingProvider::new(256)))
            .expect("open db");

        let agent = AgentId::new();
        for i in 0..5 {
            let content = format!("fact number {i} about the project");
            let emb = db.embed_text(&content).unwrap().unwrap();
            db.store(MemoryNode::new(agent, MemoryType::Semantic, content, emb))
                .unwrap();
        }

        // Must complete without panicking; counts are well-formed. On a small
        // fresh DB nothing is necessarily removed, this is a smoke test that the
        // full sweep (consolidate -> decay -> archival -> evict -> flush) runs.
        let report = run_sweep(&db);
        assert!(report.forgotten <= MAX_FORGETS_PER_SWEEP);

        drop(db);
        let _ = std::fs::remove_dir_all(&path);
    }
}
