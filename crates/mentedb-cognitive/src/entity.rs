use crate::llm::{CognitiveLlmService, EntityCandidate, EntityMergeGroup, LlmJudge};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::Path;

const SNAPSHOT_VERSION: u32 = 1;
const MIN_SUBSTRING_LEN: usize = 3;
const SUBSTRING_CONFIDENCE: f32 = 0.7;

/// Resolves entity references to canonical names using a three-tier strategy:
///
/// 1. **Learned cache** — instant lookup from alias table (no LLM call)
/// 2. **Rule-based** — case normalization, substring matching
/// 3. **LLM-powered** — CognitiveLlmService.resolve_entities() for ambiguous cases
///
/// The alias table persists across sessions so the LLM is only consulted
/// for genuinely new entity references.
#[derive(Debug, Clone)]
pub struct EntityResolver {
    /// Maps normalized alias → canonical name.
    aliases: HashMap<String, String>,
    /// Tracks confidence for each learned merge.
    confidence: HashMap<String, f32>,
}

#[derive(Serialize, Deserialize)]
struct Snapshot {
    version: u32,
    aliases: HashMap<String, String>,
    confidence: HashMap<String, f32>,
}

/// Result of resolving an entity reference.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedEntity {
    /// The canonical name for this entity.
    pub canonical: String,
    /// How confident we are in this resolution (0.0 to 1.0).
    pub confidence: f32,
    /// Whether this came from the cache, rules, or LLM.
    pub source: ResolutionSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolutionSource {
    Cache,
    RuleBased,
    Llm,
    /// No resolution found, returned as-is.
    Identity,
}

impl EntityResolver {
    pub fn new() -> Self {
        Self {
            aliases: HashMap::new(),
            confidence: HashMap::new(),
        }
    }

    /// Resolve an entity reference using the learned cache and rule-based matching.
    /// Does not call the LLM — use `resolve_with_llm` for the full pipeline.
    pub fn resolve(&self, name: &str) -> ResolvedEntity {
        let normalized = normalize_entity(name);

        // Tier 1: Exact cache hit
        if let Some(canonical) = self.aliases.get(&normalized) {
            return ResolvedEntity {
                canonical: canonical.clone(),
                confidence: self.confidence.get(&normalized).copied().unwrap_or(1.0),
                source: ResolutionSource::Cache,
            };
        }

        // Tier 2: Rule-based substring matching against known canonicals
        if let Some((canonical, conf)) = self.rule_based_match(&normalized) {
            return ResolvedEntity {
                canonical,
                confidence: conf,
                source: ResolutionSource::RuleBased,
            };
        }

        // No resolution — return as-is
        ResolvedEntity {
            canonical: normalized,
            confidence: 1.0,
            source: ResolutionSource::Identity,
        }
    }

    /// Full three-tier resolution: cache → rules → LLM.
    ///
    /// Resolves a batch of entity references. Any LLM-confirmed merges
    /// are automatically added to the alias table for future cache hits.
    pub async fn resolve_batch_with_llm<J: LlmJudge>(
        &mut self,
        names: &[String],
        contexts: &[Option<String>],
        llm: &CognitiveLlmService<J>,
    ) -> Vec<ResolvedEntity> {
        let mut results = Vec::with_capacity(names.len());
        let mut unresolved_indices = Vec::new();

        // First pass: cache + rules
        for (i, name) in names.iter().enumerate() {
            let resolved = self.resolve(name);
            if resolved.source == ResolutionSource::Identity {
                unresolved_indices.push(i);
            }
            results.push(resolved);
        }

        // If everything resolved, skip LLM
        if unresolved_indices.is_empty() {
            return results;
        }

        // Build candidates for the LLM — include ALL names for context,
        // not just unresolved ones, so the LLM can see the full picture
        let candidates: Vec<EntityCandidate> = names
            .iter()
            .enumerate()
            .map(|(i, name)| EntityCandidate {
                name: name.clone(),
                context: contexts.get(i).and_then(|c| c.clone()),
                memory_id: None,
            })
            .collect();

        if let Ok(groups) = llm.resolve_entities(&candidates).await {
            for group in &groups {
                self.learn_group(group);
            }

            // Re-resolve unresolved entries using the newly learned aliases
            for &i in &unresolved_indices {
                let re_resolved = self.resolve(&names[i]);
                if re_resolved.source == ResolutionSource::Cache {
                    results[i] = ResolvedEntity {
                        canonical: re_resolved.canonical,
                        confidence: re_resolved.confidence,
                        source: ResolutionSource::Llm,
                    };
                }
            }
        }

        results
    }

    /// Learn a merge group from the LLM, adding all aliases to the cache.
    pub fn learn_group(&mut self, group: &EntityMergeGroup) {
        let canonical = normalize_entity(&group.canonical);

        // Map canonical to itself
        self.aliases.insert(canonical.clone(), canonical.clone());
        self.confidence.insert(canonical.clone(), group.confidence);

        // Map each alias to the canonical
        for alias in &group.aliases {
            let normalized_alias = normalize_entity(alias);
            if normalized_alias != canonical {
                self.aliases
                    .insert(normalized_alias.clone(), canonical.clone());
                self.confidence.insert(normalized_alias, group.confidence);
            }
        }
    }

    /// Manually register an alias mapping.
    pub fn add_alias(&mut self, alias: &str, canonical: &str, confidence: f32) {
        let alias_norm = normalize_entity(alias);
        let canonical_norm = normalize_entity(canonical);
        self.aliases.insert(alias_norm.clone(), canonical_norm);
        self.confidence.insert(alias_norm, confidence);
    }

    /// Get the canonical name for an alias, if known.
    pub fn get_canonical(&self, name: &str) -> Option<&String> {
        let normalized = normalize_entity(name);
        self.aliases.get(&normalized)
    }

    /// Returns all known canonical entity names.
    pub fn known_entities(&self) -> Vec<String> {
        let mut entities: Vec<String> = self.aliases.values().cloned().collect();
        entities.sort();
        entities.dedup();
        entities
    }

    pub fn alias_count(&self) -> usize {
        self.aliases.len()
    }

    /// Rule-based matching: check if the input is a substring of a known
    /// canonical or vice versa. Handles "Alice" matching "Alice Smith".
    fn rule_based_match(&self, normalized: &str) -> Option<(String, f32)> {
        if normalized.len() < MIN_SUBSTRING_LEN {
            return None;
        }

        let canonicals = self.known_entities();
        for canonical in &canonicals {
            if canonical == normalized {
                continue;
            }

            // "alice" is a substring of "alice smith"
            if canonical.contains(normalized) || normalized.contains(canonical.as_str()) {
                return Some((canonical.clone(), SUBSTRING_CONFIDENCE));
            }
        }
        None
    }

    /// Save the alias table to a JSON file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let snapshot = Snapshot {
            version: SNAPSHOT_VERSION,
            aliases: self.aliases.clone(),
            confidence: self.confidence.clone(),
        };
        let json = serde_json::to_string(&snapshot)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)
    }

    /// Load the alias table from a JSON file, merging with existing entries.
    pub fn load(&mut self, path: &Path) -> io::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let snapshot: Snapshot = serde_json::from_str(&json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        if snapshot.version > SNAPSHOT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported entity snapshot version: {} (expected <= {})",
                    snapshot.version, SNAPSHOT_VERSION
                ),
            ));
        }

        for (alias, canonical) in snapshot.aliases {
            self.aliases.entry(alias).or_insert(canonical);
        }
        for (alias, conf) in snapshot.confidence {
            self.confidence.entry(alias).or_insert(conf);
        }
        Ok(())
    }
}

impl Default for EntityResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalize an entity reference: lowercase, collapse whitespace, trim.
fn normalize_entity(raw: &str) -> String {
    raw.split_whitespace()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{CognitiveLlmService, MockLlmJudge};

    #[test]
    fn test_normalize_entity() {
        assert_eq!(normalize_entity("Alice Smith"), "alice smith");
        assert_eq!(normalize_entity("  ALICE   SMITH  "), "alice smith");
        assert_eq!(normalize_entity("alice"), "alice");
    }

    #[test]
    fn test_add_alias_and_resolve() {
        let mut resolver = EntityResolver::new();
        resolver.add_alias("Alice", "Alice Smith", 0.95);
        resolver.add_alias("my manager", "Alice Smith", 0.85);

        let result = resolver.resolve("Alice");
        assert_eq!(result.canonical, "alice smith");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.source, ResolutionSource::Cache);

        let result = resolver.resolve("MY MANAGER");
        assert_eq!(result.canonical, "alice smith");
        assert_eq!(result.source, ResolutionSource::Cache);
    }

    #[test]
    fn test_unresolved_returns_identity() {
        let resolver = EntityResolver::new();
        let result = resolver.resolve("Bob");
        assert_eq!(result.canonical, "bob");
        assert_eq!(result.source, ResolutionSource::Identity);
    }

    #[test]
    fn test_rule_based_substring_match() {
        let mut resolver = EntityResolver::new();
        resolver.add_alias("alice smith", "alice smith", 1.0);

        // "alice" is a substring of "alice smith"
        let result = resolver.resolve("Alice");
        assert_eq!(result.canonical, "alice smith");
        assert_eq!(result.source, ResolutionSource::RuleBased);
        assert_eq!(result.confidence, SUBSTRING_CONFIDENCE);
    }

    #[test]
    fn test_short_names_skip_substring_match() {
        let mut resolver = EntityResolver::new();
        resolver.add_alias("db", "database", 1.0);

        // "db" is too short for substring matching (< 3 chars)
        let result = resolver.resolve("db");
        // Should hit cache directly since we added it as an alias
        assert_eq!(result.canonical, "database");
        assert_eq!(result.source, ResolutionSource::Cache);
    }

    #[test]
    fn test_learn_group() {
        let mut resolver = EntityResolver::new();
        resolver.learn_group(&EntityMergeGroup {
            canonical: "Alice Smith".to_string(),
            aliases: vec!["Alice".to_string(), "my manager".to_string()],
            confidence: 0.9,
        });

        assert_eq!(
            resolver.get_canonical("alice"),
            Some(&"alice smith".to_string())
        );
        assert_eq!(
            resolver.get_canonical("my manager"),
            Some(&"alice smith".to_string())
        );
        assert_eq!(
            resolver.get_canonical("alice smith"),
            Some(&"alice smith".to_string())
        );
    }

    #[test]
    fn test_known_entities() {
        let mut resolver = EntityResolver::new();
        resolver.learn_group(&EntityMergeGroup {
            canonical: "Alice Smith".to_string(),
            aliases: vec!["Alice".to_string()],
            confidence: 0.9,
        });
        resolver.add_alias("postgres", "PostgreSQL", 1.0);

        let entities = resolver.known_entities();
        assert!(entities.contains(&"alice smith".to_string()));
        assert!(entities.contains(&"postgresql".to_string()));
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entities.json");

        let mut resolver = EntityResolver::new();
        resolver.learn_group(&EntityMergeGroup {
            canonical: "Alice Smith".to_string(),
            aliases: vec!["Alice".to_string(), "my manager".to_string()],
            confidence: 0.9,
        });
        resolver.save(&path).unwrap();

        let mut loaded = EntityResolver::new();
        loaded.load(&path).unwrap();

        assert_eq!(
            loaded.get_canonical("alice"),
            Some(&"alice smith".to_string())
        );
        assert_eq!(
            loaded.get_canonical("my manager"),
            Some(&"alice smith".to_string())
        );
    }

    #[test]
    fn test_load_merges_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entities.json");

        let mut resolver1 = EntityResolver::new();
        resolver1.add_alias("alice", "alice smith", 0.9);
        resolver1.save(&path).unwrap();

        let mut resolver2 = EntityResolver::new();
        resolver2.add_alias("bob", "bob jones", 0.8);
        resolver2.load(&path).unwrap();

        // Both aliases should exist
        assert_eq!(
            resolver2.get_canonical("alice"),
            Some(&"alice smith".to_string())
        );
        assert_eq!(
            resolver2.get_canonical("bob"),
            Some(&"bob jones".to_string())
        );
    }

    #[tokio::test]
    async fn test_resolve_batch_with_llm() {
        let judge = MockLlmJudge::new(
            r#"{"groups": [{"canonical": "Alice Smith", "aliases": ["Alice", "my manager"], "confidence": 0.9}]}"#,
        );
        let llm = CognitiveLlmService::new(judge);
        let mut resolver = EntityResolver::new();

        let names = vec![
            "Alice".to_string(),
            "my manager".to_string(),
            "Bob".to_string(),
        ];
        let contexts = vec![None, Some("Alice is my manager".to_string()), None];

        let results = resolver
            .resolve_batch_with_llm(&names, &contexts, &llm)
            .await;

        // Alice and my manager should resolve to alice smith via LLM
        assert_eq!(results[0].canonical, "alice smith");
        assert_eq!(results[0].source, ResolutionSource::Llm);
        assert_eq!(results[1].canonical, "alice smith");
        assert_eq!(results[1].source, ResolutionSource::Llm);

        // Bob wasn't in the LLM response, stays identity
        assert_eq!(results[2].canonical, "bob");
        assert_eq!(results[2].source, ResolutionSource::Identity);

        // Aliases should be cached for next time
        assert_eq!(resolver.alias_count(), 3); // alice smith, alice, my manager
    }

    #[tokio::test]
    async fn test_resolve_batch_skips_llm_when_cached() {
        let judge = MockLlmJudge::new(r#"{"groups": []}"#);
        let llm = CognitiveLlmService::new(judge);
        let mut resolver = EntityResolver::new();

        // Pre-teach the cache
        resolver.add_alias("alice", "alice smith", 0.95);

        let names = vec!["Alice".to_string()];
        let contexts = vec![None];

        let results = resolver
            .resolve_batch_with_llm(&names, &contexts, &llm)
            .await;

        // Should come from cache, not LLM
        assert_eq!(results[0].canonical, "alice smith");
        assert_eq!(results[0].source, ResolutionSource::Cache);
    }
}
