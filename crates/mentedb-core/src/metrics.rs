//! Observability metrics for MenteDB.

use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic counters for all MenteDB operations.
pub struct Metrics {
    /// Total write operations.
    pub writes_total: AtomicU64,
    /// Total read operations.
    pub reads_total: AtomicU64,
    /// Total delete operations.
    pub deletes_total: AtomicU64,
    /// Cache hits.
    pub cache_hits: AtomicU64,
    /// Cache misses.
    pub cache_misses: AtomicU64,
    /// WAL sync operations.
    pub wal_syncs: AtomicU64,
    /// Context assembly operations.
    pub context_assemblies: AtomicU64,
    /// MQL queries parsed.
    pub mql_queries_parsed: AtomicU64,
    /// Contradictions detected.
    pub contradictions_detected: AtomicU64,
    /// Beliefs propagated.
    pub beliefs_propagated: AtomicU64,
    /// Speculative cache hits.
    pub speculative_hits: AtomicU64,
    /// Speculative cache misses.
    pub speculative_misses: AtomicU64,
    /// Sum of write latencies in microseconds.
    pub write_latency_us_sum: AtomicU64,
    /// Number of write latency samples.
    pub write_latency_count: AtomicU64,
    /// Sum of read latencies in microseconds.
    pub read_latency_us_sum: AtomicU64,
    /// Number of read latency samples.
    pub read_latency_count: AtomicU64,
}

/// A point-in-time snapshot of metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Total write operations.
    pub writes_total: u64,
    /// Total read operations.
    pub reads_total: u64,
    /// Total delete operations.
    pub deletes_total: u64,
    /// Cache hit rate (0.0–1.0).
    pub cache_hit_rate: f64,
    /// Average write latency in microseconds.
    pub avg_write_latency_us: f64,
    /// Average read latency in microseconds.
    pub avg_read_latency_us: f64,
    /// Total contradictions detected.
    pub contradictions_detected: u64,
    /// Speculative cache hit rate (0.0–1.0).
    pub speculative_hit_rate: f64,
}

impl Metrics {
    /// Create a new zeroed metrics instance.
    pub fn new() -> Self {
        Self {
            writes_total: AtomicU64::new(0),
            reads_total: AtomicU64::new(0),
            deletes_total: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            wal_syncs: AtomicU64::new(0),
            context_assemblies: AtomicU64::new(0),
            mql_queries_parsed: AtomicU64::new(0),
            contradictions_detected: AtomicU64::new(0),
            beliefs_propagated: AtomicU64::new(0),
            speculative_hits: AtomicU64::new(0),
            speculative_misses: AtomicU64::new(0),
            write_latency_us_sum: AtomicU64::new(0),
            write_latency_count: AtomicU64::new(0),
            read_latency_us_sum: AtomicU64::new(0),
            read_latency_count: AtomicU64::new(0),
        }
    }

    /// Increment writes counter.
    pub fn inc_writes(&self) {
        self.writes_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment reads counter.
    pub fn inc_reads(&self) {
        self.reads_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment deletes counter.
    pub fn inc_deletes(&self) {
        self.deletes_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment cache hits counter.
    pub fn inc_cache_hits(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment cache misses counter.
    pub fn inc_cache_misses(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a write latency sample.
    pub fn record_write_latency(&self, microseconds: u64) {
        self.write_latency_us_sum
            .fetch_add(microseconds, Ordering::Relaxed);
        self.write_latency_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a read latency sample.
    pub fn record_read_latency(&self, microseconds: u64) {
        self.read_latency_us_sum
            .fetch_add(microseconds, Ordering::Relaxed);
        self.read_latency_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Export metrics in Prometheus text exposition format.
    pub fn export_prometheus(&self) -> String {
        let mut out = String::with_capacity(2048);

        let counters = [
            ("mentedb_writes_total", "Total write operations", &self.writes_total),
            ("mentedb_reads_total", "Total read operations", &self.reads_total),
            ("mentedb_deletes_total", "Total delete operations", &self.deletes_total),
            ("mentedb_cache_hits_total", "Total cache hits", &self.cache_hits),
            ("mentedb_cache_misses_total", "Total cache misses", &self.cache_misses),
            ("mentedb_wal_syncs_total", "Total WAL sync operations", &self.wal_syncs),
            ("mentedb_context_assemblies_total", "Total context assemblies", &self.context_assemblies),
            ("mentedb_mql_queries_parsed_total", "Total MQL queries parsed", &self.mql_queries_parsed),
            ("mentedb_contradictions_detected_total", "Total contradictions detected", &self.contradictions_detected),
            ("mentedb_beliefs_propagated_total", "Total beliefs propagated", &self.beliefs_propagated),
            ("mentedb_speculative_hits_total", "Speculative cache hits", &self.speculative_hits),
            ("mentedb_speculative_misses_total", "Speculative cache misses", &self.speculative_misses),
        ];

        for (name, help, counter) in &counters {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {}\n",
                counter.load(Ordering::Relaxed)
            ));
        }

        // Latency summaries
        let wl_sum = self.write_latency_us_sum.load(Ordering::Relaxed);
        let wl_count = self.write_latency_count.load(Ordering::Relaxed);
        out.push_str(&format!(
            "# HELP mentedb_write_latency_us Write latency in microseconds\n\
             # TYPE mentedb_write_latency_us summary\n\
             mentedb_write_latency_us_sum {wl_sum}\n\
             mentedb_write_latency_us_count {wl_count}\n"
        ));

        let rl_sum = self.read_latency_us_sum.load(Ordering::Relaxed);
        let rl_count = self.read_latency_count.load(Ordering::Relaxed);
        out.push_str(&format!(
            "# HELP mentedb_read_latency_us Read latency in microseconds\n\
             # TYPE mentedb_read_latency_us summary\n\
             mentedb_read_latency_us_sum {rl_sum}\n\
             mentedb_read_latency_us_count {rl_count}\n"
        ));

        out
    }

    /// Export metrics as a JSON string.
    pub fn export_json(&self) -> String {
        let snap = self.snapshot();
        format!(
            concat!(
                "{{",
                "\"writes_total\":{},",
                "\"reads_total\":{},",
                "\"deletes_total\":{},",
                "\"cache_hit_rate\":{:.4},",
                "\"avg_write_latency_us\":{:.2},",
                "\"avg_read_latency_us\":{:.2},",
                "\"contradictions_detected\":{},",
                "\"speculative_hit_rate\":{:.4},",
                "\"wal_syncs\":{},",
                "\"context_assemblies\":{},",
                "\"mql_queries_parsed\":{},",
                "\"beliefs_propagated\":{}",
                "}}"
            ),
            snap.writes_total,
            snap.reads_total,
            snap.deletes_total,
            snap.cache_hit_rate,
            snap.avg_write_latency_us,
            snap.avg_read_latency_us,
            snap.contradictions_detected,
            snap.speculative_hit_rate,
            self.wal_syncs.load(Ordering::Relaxed),
            self.context_assemblies.load(Ordering::Relaxed),
            self.mql_queries_parsed.load(Ordering::Relaxed),
            self.beliefs_propagated.load(Ordering::Relaxed),
        )
    }

    /// Take a point-in-time snapshot of all metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let cache_total = hits + misses;
        let cache_hit_rate = if cache_total > 0 {
            hits as f64 / cache_total as f64
        } else {
            0.0
        };

        let wl_sum = self.write_latency_us_sum.load(Ordering::Relaxed);
        let wl_count = self.write_latency_count.load(Ordering::Relaxed);
        let avg_write = if wl_count > 0 {
            wl_sum as f64 / wl_count as f64
        } else {
            0.0
        };

        let rl_sum = self.read_latency_us_sum.load(Ordering::Relaxed);
        let rl_count = self.read_latency_count.load(Ordering::Relaxed);
        let avg_read = if rl_count > 0 {
            rl_sum as f64 / rl_count as f64
        } else {
            0.0
        };

        let spec_hits = self.speculative_hits.load(Ordering::Relaxed);
        let spec_misses = self.speculative_misses.load(Ordering::Relaxed);
        let spec_total = spec_hits + spec_misses;
        let speculative_hit_rate = if spec_total > 0 {
            spec_hits as f64 / spec_total as f64
        } else {
            0.0
        };

        MetricsSnapshot {
            writes_total: self.writes_total.load(Ordering::Relaxed),
            reads_total: self.reads_total.load(Ordering::Relaxed),
            deletes_total: self.deletes_total.load(Ordering::Relaxed),
            cache_hit_rate,
            avg_write_latency_us: avg_write,
            avg_read_latency_us: avg_read,
            contradictions_detected: self.contradictions_detected.load(Ordering::Relaxed),
            speculative_hit_rate,
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn increment_and_read() {
        let m = Metrics::new();
        m.inc_writes();
        m.inc_writes();
        m.inc_reads();
        m.inc_cache_hits();
        m.inc_cache_hits();
        m.inc_cache_misses();
        m.record_write_latency(100);
        m.record_write_latency(200);

        let snap = m.snapshot();
        assert_eq!(snap.writes_total, 2);
        assert_eq!(snap.reads_total, 1);
        assert!((snap.cache_hit_rate - 2.0 / 3.0).abs() < 0.01);
        assert!((snap.avg_write_latency_us - 150.0).abs() < 0.01);
    }

    #[test]
    fn prometheus_format() {
        let m = Metrics::new();
        m.inc_writes();
        m.inc_reads();
        m.inc_reads();

        let prom = m.export_prometheus();
        assert!(prom.contains("# HELP mentedb_writes_total Total write operations"));
        assert!(prom.contains("# TYPE mentedb_writes_total counter"));
        assert!(prom.contains("mentedb_writes_total 1"));
        assert!(prom.contains("mentedb_reads_total 2"));
    }

    #[test]
    fn json_format() {
        let m = Metrics::new();
        m.inc_writes();
        m.inc_deletes();
        m.record_read_latency(500);

        let json = m.export_json();
        // Verify it's valid JSON by checking structure
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("\"writes_total\":1"));
        assert!(json.contains("\"deletes_total\":1"));
        assert!(json.contains("\"avg_read_latency_us\":500.00"));
    }
}
