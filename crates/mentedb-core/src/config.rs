//! Configuration types for MenteDB.

use serde::Deserialize;
use std::path::Path;

use crate::error::{MenteError, MenteResult};

/// Top-level configuration for a MenteDB instance.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct MenteConfig {
    /// Storage engine configuration.
    #[serde(default)]
    pub storage: StorageConfig,
    /// Index layer configuration.
    #[serde(default)]
    pub index: IndexConfig,
    /// Context assembly configuration.
    #[serde(default)]
    pub context: ContextConfig,
    /// Cognitive engine configuration.
    #[serde(default)]
    pub cognitive: CognitiveConfig,
    /// Memory consolidation configuration.
    #[serde(default)]
    pub consolidation: ConsolidationConfig,
    /// Server configuration.
    #[serde(default)]
    pub server: ServerConfig,
}

/// Storage engine settings.
#[derive(Debug, Clone, Deserialize)]
pub struct StorageConfig {
    /// Directory for data files.
    pub data_dir: String,
    /// Number of pages in the buffer pool.
    #[serde(default = "default_buffer_pool_size")]
    pub buffer_pool_size: usize,
    /// Page size in bytes.
    #[serde(default = "default_page_size")]
    pub page_size: usize,
}

fn default_buffer_pool_size() -> usize {
    1024
}
fn default_page_size() -> usize {
    16384
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: "data".to_string(),
            buffer_pool_size: default_buffer_pool_size(),
            page_size: default_page_size(),
        }
    }
}

/// HNSW index settings.
#[derive(Debug, Clone, Deserialize)]
pub struct IndexConfig {
    /// Number of bidirectional links per node.
    #[serde(default = "default_hnsw_m")]
    pub hnsw_m: usize,
    /// Size of the dynamic candidate list during construction.
    #[serde(default = "default_hnsw_ef_construction")]
    pub hnsw_ef_construction: usize,
    /// Size of the dynamic candidate list during search.
    #[serde(default = "default_hnsw_ef_search")]
    pub hnsw_ef_search: usize,
}

fn default_hnsw_m() -> usize {
    16
}
fn default_hnsw_ef_construction() -> usize {
    200
}
fn default_hnsw_ef_search() -> usize {
    50
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            hnsw_m: default_hnsw_m(),
            hnsw_ef_construction: default_hnsw_ef_construction(),
            hnsw_ef_search: default_hnsw_ef_search(),
        }
    }
}

/// Context assembly settings.
#[derive(Debug, Clone, Deserialize)]
pub struct ContextConfig {
    /// Default token budget for context windows.
    #[serde(default = "default_token_budget")]
    pub default_token_budget: usize,
    /// Multiplier for estimating token counts from word counts.
    #[serde(default = "default_token_multiplier")]
    pub token_multiplier: f32,
    /// Fraction of budget for the system zone.
    #[serde(default = "default_zone_system_pct")]
    pub zone_system_pct: f32,
    /// Fraction of budget for the critical zone.
    #[serde(default = "default_zone_critical_pct")]
    pub zone_critical_pct: f32,
    /// Fraction of budget for the primary zone.
    #[serde(default = "default_zone_primary_pct")]
    pub zone_primary_pct: f32,
    /// Fraction of budget for the supporting zone.
    #[serde(default = "default_zone_supporting_pct")]
    pub zone_supporting_pct: f32,
    /// Fraction of budget for the reference zone.
    #[serde(default = "default_zone_reference_pct")]
    pub zone_reference_pct: f32,
}

fn default_token_budget() -> usize {
    4096
}
fn default_token_multiplier() -> f32 {
    1.3
}
fn default_zone_system_pct() -> f32 {
    0.10
}
fn default_zone_critical_pct() -> f32 {
    0.25
}
fn default_zone_primary_pct() -> f32 {
    0.35
}
fn default_zone_supporting_pct() -> f32 {
    0.20
}
fn default_zone_reference_pct() -> f32 {
    0.10
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            default_token_budget: default_token_budget(),
            token_multiplier: default_token_multiplier(),
            zone_system_pct: default_zone_system_pct(),
            zone_critical_pct: default_zone_critical_pct(),
            zone_primary_pct: default_zone_primary_pct(),
            zone_supporting_pct: default_zone_supporting_pct(),
            zone_reference_pct: default_zone_reference_pct(),
        }
    }
}

/// Cognitive engine settings.
#[derive(Debug, Clone, Deserialize)]
pub struct CognitiveConfig {
    /// Similarity threshold above which memories are considered contradictory.
    #[serde(default = "default_contradiction_threshold")]
    pub contradiction_threshold: f32,
    /// Minimum similarity for memories to be considered related.
    #[serde(default = "default_related_threshold_min")]
    pub related_threshold_min: f32,
    /// Maximum similarity for the "related" band (above this is near-duplicate).
    #[serde(default = "default_related_threshold_max")]
    pub related_threshold_max: f32,
    /// Similarity threshold for interference detection.
    #[serde(default = "default_interference_threshold")]
    pub interference_threshold: f32,
    /// Number of entries in the speculative pre-assembly cache.
    #[serde(default = "default_speculative_cache_size")]
    pub speculative_cache_size: usize,
    /// Hit rate threshold for the speculative cache to remain active.
    #[serde(default = "default_speculative_hit_threshold")]
    pub speculative_hit_threshold: f32,
    /// Maximum number of turns to track in a trajectory.
    #[serde(default = "default_max_trajectory_turns")]
    pub max_trajectory_turns: usize,
    /// Maximum number of active pain signal warnings.
    #[serde(default = "default_max_pain_warnings")]
    pub max_pain_warnings: usize,
    /// Maximum number of active phantom memory warnings.
    #[serde(default = "default_max_phantom_warnings")]
    pub max_phantom_warnings: usize,
}

fn default_contradiction_threshold() -> f32 {
    0.95
}
fn default_related_threshold_min() -> f32 {
    0.6
}
fn default_related_threshold_max() -> f32 {
    0.85
}
fn default_interference_threshold() -> f32 {
    0.8
}
fn default_speculative_cache_size() -> usize {
    10
}
fn default_speculative_hit_threshold() -> f32 {
    0.5
}
fn default_max_trajectory_turns() -> usize {
    100
}
fn default_max_pain_warnings() -> usize {
    5
}
fn default_max_phantom_warnings() -> usize {
    5
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        Self {
            contradiction_threshold: default_contradiction_threshold(),
            related_threshold_min: default_related_threshold_min(),
            related_threshold_max: default_related_threshold_max(),
            interference_threshold: default_interference_threshold(),
            speculative_cache_size: default_speculative_cache_size(),
            speculative_hit_threshold: default_speculative_hit_threshold(),
            max_trajectory_turns: default_max_trajectory_turns(),
            max_pain_warnings: default_max_pain_warnings(),
            max_phantom_warnings: default_max_phantom_warnings(),
        }
    }
}

/// Memory consolidation settings.
#[derive(Debug, Clone, Deserialize)]
pub struct ConsolidationConfig {
    /// Half-life for temporal salience decay, in hours.
    #[serde(default = "default_decay_half_life_hours")]
    pub decay_half_life_hours: f64,
    /// Minimum salience before a memory is eligible for archival.
    #[serde(default = "default_min_salience")]
    pub min_salience: f32,
    /// Minimum age in days before a memory can be archived.
    #[serde(default = "default_archival_min_age_days")]
    pub archival_min_age_days: u64,
    /// Maximum salience for archival eligibility.
    #[serde(default = "default_archival_max_salience")]
    pub archival_max_salience: f32,
}

fn default_decay_half_life_hours() -> f64 {
    168.0
}
fn default_min_salience() -> f32 {
    0.01
}
fn default_archival_min_age_days() -> u64 {
    30
}
fn default_archival_max_salience() -> f32 {
    0.05
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            decay_half_life_hours: default_decay_half_life_hours(),
            min_salience: default_min_salience(),
            archival_min_age_days: default_archival_min_age_days(),
            archival_max_salience: default_archival_max_salience(),
        }
    }
}

/// Server settings.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Bind address.
    #[serde(default = "default_host")]
    pub host: String,
    /// Listen port.
    #[serde(default = "default_port")]
    pub port: u16,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    6677
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

impl MenteConfig {
    /// Load configuration from a JSON file.
    pub fn from_file(path: &Path) -> MenteResult<Self> {
        let contents = std::fs::read_to_string(path).map_err(MenteError::Io)?;
        serde_json::from_str(&contents).map_err(|e| MenteError::Serialization(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let cfg = MenteConfig::default();

        assert_eq!(cfg.storage.data_dir, "data");
        assert_eq!(cfg.storage.buffer_pool_size, 1024);
        assert_eq!(cfg.storage.page_size, 16384);

        assert_eq!(cfg.index.hnsw_m, 16);
        assert_eq!(cfg.index.hnsw_ef_construction, 200);
        assert_eq!(cfg.index.hnsw_ef_search, 50);

        assert_eq!(cfg.context.default_token_budget, 4096);
        assert!((cfg.context.token_multiplier - 1.3).abs() < f32::EPSILON);
        assert!((cfg.context.zone_system_pct - 0.10).abs() < f32::EPSILON);
        assert!((cfg.context.zone_critical_pct - 0.25).abs() < f32::EPSILON);
        assert!((cfg.context.zone_primary_pct - 0.35).abs() < f32::EPSILON);
        assert!((cfg.context.zone_supporting_pct - 0.20).abs() < f32::EPSILON);
        assert!((cfg.context.zone_reference_pct - 0.10).abs() < f32::EPSILON);

        assert!((cfg.cognitive.contradiction_threshold - 0.95).abs() < f32::EPSILON);
        assert!((cfg.cognitive.related_threshold_min - 0.6).abs() < f32::EPSILON);
        assert!((cfg.cognitive.related_threshold_max - 0.85).abs() < f32::EPSILON);
        assert!((cfg.cognitive.interference_threshold - 0.8).abs() < f32::EPSILON);
        assert_eq!(cfg.cognitive.speculative_cache_size, 10);
        assert!((cfg.cognitive.speculative_hit_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.cognitive.max_trajectory_turns, 100);
        assert_eq!(cfg.cognitive.max_pain_warnings, 5);
        assert_eq!(cfg.cognitive.max_phantom_warnings, 5);

        assert!((cfg.consolidation.decay_half_life_hours - 168.0).abs() < f64::EPSILON);
        assert!((cfg.consolidation.min_salience - 0.01).abs() < f32::EPSILON);
        assert_eq!(cfg.consolidation.archival_min_age_days, 30);
        assert!((cfg.consolidation.archival_max_salience - 0.05).abs() < f32::EPSILON);

        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.server.port, 6677);
    }

    #[test]
    fn test_from_file() {
        let dir = std::env::temp_dir().join("mentedb_config_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");

        let json = r#"{
            "storage": {
                "data_dir": "/var/mentedb",
                "buffer_pool_size": 2048,
                "page_size": 8192
            },
            "index": {
                "hnsw_m": 32,
                "hnsw_ef_construction": 400,
                "hnsw_ef_search": 100
            },
            "server": {
                "host": "127.0.0.1",
                "port": 9999
            }
        }"#;
        std::fs::write(&path, json).unwrap();

        let cfg = MenteConfig::from_file(&path).unwrap();

        assert_eq!(cfg.storage.data_dir, "/var/mentedb");
        assert_eq!(cfg.storage.buffer_pool_size, 2048);
        assert_eq!(cfg.storage.page_size, 8192);
        assert_eq!(cfg.index.hnsw_m, 32);
        assert_eq!(cfg.index.hnsw_ef_construction, 400);
        assert_eq!(cfg.index.hnsw_ef_search, 100);
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert_eq!(cfg.server.port, 9999);

        // Sections not provided should use defaults.
        assert_eq!(cfg.context.default_token_budget, 4096);
        assert!((cfg.cognitive.contradiction_threshold - 0.95).abs() < f32::EPSILON);
        assert!((cfg.consolidation.decay_half_life_hours - 168.0).abs() < f64::EPSILON);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_from_file_empty_object() {
        let dir = std::env::temp_dir().join("mentedb_config_test_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, "{}").unwrap();

        let cfg = MenteConfig::from_file(&path).unwrap();
        let defaults = MenteConfig::default();

        assert_eq!(
            cfg.storage.buffer_pool_size,
            defaults.storage.buffer_pool_size
        );
        assert_eq!(cfg.index.hnsw_m, defaults.index.hnsw_m);
        assert_eq!(cfg.server.port, defaults.server.port);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_from_file_not_found() {
        let result = MenteConfig::from_file(Path::new("/nonexistent/config.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_from_file_invalid_json() {
        let dir = std::env::temp_dir().join("mentedb_config_test_invalid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, "not json at all").unwrap();

        let result = MenteConfig::from_file(&path);
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }
}
