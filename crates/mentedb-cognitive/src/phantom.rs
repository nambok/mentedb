use std::collections::HashSet;

use ahash::AHashSet;
use mentedb_core::types::Timestamp;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PhantomPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhantomMemory {
    pub id: Uuid,
    pub gap_description: String,
    pub source_reference: String,
    pub source_turn: u64,
    pub priority: PhantomPriority,
    pub created_at: Timestamp,
    pub resolved: bool,
}

/// Explicit registry of known entities. The AI client registers entities it
/// cares about so that gap detection can be precise rather than heuristic.
pub struct EntityRegistry {
    known_entities: AHashSet<String>,
}

impl EntityRegistry {
    pub fn new() -> Self {
        Self {
            known_entities: AHashSet::new(),
        }
    }

    pub fn register(&mut self, entity: &str) {
        self.known_entities.insert(entity.to_string());
    }

    pub fn register_batch(&mut self, entities: &[&str]) {
        for entity in entities {
            self.known_entities.insert((*entity).to_string());
        }
    }

    pub fn unregister(&mut self, entity: &str) {
        self.known_entities.remove(entity);
    }

    pub fn is_known(&self, entity: &str) -> bool {
        self.known_entities.contains(entity)
    }

    pub fn list(&self) -> Vec<&str> {
        self.known_entities.iter().map(|s| s.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.known_entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.known_entities.is_empty()
    }
}

impl Default for EntityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for phantom memory detection.
#[derive(Debug, Clone)]
pub struct PhantomConfig {
    /// Words to skip when detecting capitalized entity references.
    pub stop_words: HashSet<String>,
    /// Maximum number of warnings to include in formatted output.
    pub max_warnings: usize,
    /// Minimum word length for technical term detection.
    pub min_word_length: usize,
    /// Whether to use heuristic (capitalized word) detection as a fallback.
    /// Default `true` for backward compatibility; set to `false` to rely
    /// solely on the explicit entity registry.
    pub use_heuristic_detection: bool,
}

impl Default for PhantomConfig {
    fn default() -> Self {
        let stop_words: HashSet<String> = [
            "the", "a", "an", "is", "are", "was", "were", "it", "this", "that",
            "we", "you", "they", "he", "she", "i", "my", "our", "but", "and",
            "or", "if", "then", "when", "how", "what", "where", "who", "do",
            "does", "did", "have", "has", "had", "be", "been", "being", "so",
            "also", "just", "very", "too", "not", "no", "yes", "can", "will",
            "should", "would", "could", "may", "might", "must", "shall",
            "note", "see", "use", "make", "let", "new", "set", "get",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        Self {
            stop_words,
            max_warnings: 5,
            min_word_length: 3,
            use_heuristic_detection: true,
        }
    }
}

pub struct PhantomTracker {
    phantoms: Vec<PhantomMemory>,
    config: PhantomConfig,
    entity_registry: EntityRegistry,
}

impl PhantomTracker {
    pub fn new(config: PhantomConfig) -> Self {
        Self {
            phantoms: Vec::new(),
            config,
            entity_registry: EntityRegistry::new(),
        }
    }

    /// Convenience: register an entity in the embedded registry.
    pub fn register_entity(&mut self, entity: &str) {
        self.entity_registry.register(entity);
    }

    /// Convenience: register multiple entities at once.
    pub fn register_entities(&mut self, entities: &[&str]) {
        self.entity_registry.register_batch(entities);
    }

    /// Return a reference to the entity registry.
    pub fn entity_registry(&self) -> &EntityRegistry {
        &self.entity_registry
    }

    /// Return a mutable reference to the entity registry.
    pub fn entity_registry_mut(&mut self) -> &mut EntityRegistry {
        &mut self.entity_registry
    }

    /// Scan content for entity references not in `known_entities`.
    ///
    /// Two detection strategies are combined:
    /// 1. **Registry-based** (primary): any registered entity found in
    ///    `content` that is NOT in `known_entities` is flagged at
    ///    `Medium`/`High` priority.
    /// 2. **Heuristic** (fallback): capitalized words, quoted terms, and
    ///    technical terms are flagged at `Low` priority. Disabled when
    ///    `PhantomConfig::use_heuristic_detection` is `false`.
    pub fn detect_gaps(
        &mut self,
        content: &str,
        known_entities: &[String],
        turn_id: u64,
    ) -> Vec<PhantomMemory> {
        let known_lower: Vec<String> = known_entities.iter().map(|e| e.to_lowercase()).collect();
        let mut detected: Vec<(String, PhantomPriority)> = Vec::new();
        let mut seen = AHashSet::new();

        // --- Primary: registry-based detection ---
        let content_lower = content.to_lowercase();
        for entity in self.entity_registry.list() {
            let entity_lower = entity.to_lowercase();
            if content_lower.contains(&entity_lower)
                && !known_lower.contains(&entity_lower)
                && seen.insert(entity_lower)
            {
                let priority = if entity.split_whitespace().count() > 1 {
                    PhantomPriority::High
                } else {
                    PhantomPriority::Medium
                };
                detected.push((entity.to_string(), priority));
            }
        }

        // --- Fallback: heuristic detection (only when enabled) ---
        if self.config.use_heuristic_detection {
            let heuristic = self.heuristic_detect(content, &known_lower, &mut seen);
            for entity in heuristic {
                detected.push((entity, PhantomPriority::Low));
            }
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let new_phantoms: Vec<PhantomMemory> = detected
            .into_iter()
            .map(|(entity, priority)| PhantomMemory {
                id: Uuid::new_v4(),
                gap_description: format!("No stored knowledge about '{}'", entity),
                source_reference: entity,
                source_turn: turn_id,
                priority,
                created_at: now,
                resolved: false,
            })
            .collect();

        self.phantoms.extend(new_phantoms.clone());
        new_phantoms
    }

    /// Preferred explicit API: the caller provides exactly which entities
    /// were mentioned. Entities not in the registry are flagged as gaps.
    pub fn detect_gaps_explicit(
        &mut self,
        content: &str,
        mentioned_entities: &[&str],
        turn_id: u64,
    ) -> Vec<PhantomMemory> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let _ = content; // available for future use; presence keeps API uniform

        let new_phantoms: Vec<PhantomMemory> = mentioned_entities
            .iter()
            .filter(|e| !self.entity_registry.is_known(e))
            .map(|entity| PhantomMemory {
                id: Uuid::new_v4(),
                gap_description: format!("No stored knowledge about '{}'", entity),
                source_reference: (*entity).to_string(),
                source_turn: turn_id,
                priority: PhantomPriority::Medium,
                created_at: now,
                resolved: false,
            })
            .collect();

        self.phantoms.extend(new_phantoms.clone());
        new_phantoms
    }

    /// Heuristic entity detection (capitalized words, quotes, technical terms).
    fn heuristic_detect(
        &self,
        content: &str,
        known_lower: &[String],
        seen: &mut AHashSet<String>,
    ) -> Vec<String> {
        let mut detected = Vec::new();

        // Detect quoted terms
        let mut in_quote = false;
        let mut quote_start = 0;
        for (i, ch) in content.char_indices() {
            if ch == '\'' || ch == '"' || ch == '\u{2018}' || ch == '\u{2019}' {
                if in_quote {
                    let term = &content[quote_start..i];
                    let trimmed = term.trim();
                    if !trimmed.is_empty()
                        && trimmed.len() >= 2
                        && !known_lower.contains(&trimmed.to_lowercase())
                        && seen.insert(trimmed.to_lowercase())
                    {
                        detected.push(trimmed.to_string());
                    }
                    in_quote = false;
                } else {
                    in_quote = true;
                    quote_start = i + ch.len_utf8();
                }
            }
        }

        // Detect capitalized sequences (e.g., "Kubernetes Cluster", "JWT Token")
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut i = 0;
        while i < words.len() {
            let w = words[i].trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_' && c != '.');
            if !w.is_empty() && w.chars().next().is_some_and(|c| c.is_uppercase()) && w.len() >= 2 {
                let mut entity_parts = vec![w.to_string()];
                let mut j = i + 1;
                while j < words.len() {
                    let next = words[j].trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_' && c != '.');
                    if !next.is_empty() && next.chars().next().is_some_and(|c| c.is_uppercase()) {
                        entity_parts.push(next.to_string());
                        j += 1;
                    } else {
                        break;
                    }
                }

                let entity = entity_parts.join(" ");
                if entity.split_whitespace().count() == 1
                    && self.config.stop_words.contains(&entity.to_lowercase())
                {
                    i = j;
                    continue;
                }

                if !known_lower.contains(&entity.to_lowercase())
                    && seen.insert(entity.to_lowercase())
                {
                    detected.push(entity);
                }
                i = j;
            } else {
                if !w.is_empty() && w.len() >= self.config.min_word_length {
                    let is_technical = w.contains('-') || w.contains('.') || w.contains('_')
                        || (w.len() >= self.config.min_word_length && w.chars().all(|c| c.is_uppercase() || c.is_ascii_digit() || c == '_'));

                    if is_technical
                        && !known_lower.contains(&w.to_lowercase())
                        && seen.insert(w.to_lowercase())
                    {
                        detected.push(w.to_string());
                    }
                }
                i += 1;
            }
        }

        detected
    }

    pub fn resolve(&mut self, phantom_id: Uuid) {
        if let Some(p) = self.phantoms.iter_mut().find(|p| p.id == phantom_id) {
            p.resolved = true;
        }
    }

    pub fn get_active_phantoms(&self) -> Vec<&PhantomMemory> {
        let mut active: Vec<&PhantomMemory> = self.phantoms.iter().filter(|p| !p.resolved).collect();
        active.sort_by(|a, b| b.priority.cmp(&a.priority));
        active
    }

    pub fn format_phantom_warnings(&self) -> String {
        let active = self.get_active_phantoms();
        if active.is_empty() {
            return String::new();
        }

        active
            .iter()
            .take(self.config.max_warnings)
            .map(|p| {
                format!(
                    "WARNING: User referenced '{}' but no details stored. Consider asking.",
                    p.source_reference
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for PhantomTracker {
    fn default() -> Self {
        Self::new(PhantomConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_unknown_entities() {
        let mut tracker = PhantomTracker::default();
        let known = vec!["React".to_string(), "TypeScript".to_string()];
        let phantoms = tracker.detect_gaps(
            "We need to deploy to the Kubernetes cluster using Terraform",
            &known,
            1,
        );
        let refs: Vec<&str> = phantoms.iter().map(|p| p.source_reference.as_str()).collect();
        assert!(refs.iter().any(|r| r.contains("Kubernetes")), "Expected Kubernetes, got: {:?}", refs);
        assert!(refs.iter().any(|r| r.contains("Terraform")), "Expected Terraform, got: {:?}", refs);
    }

    #[test]
    fn test_resolve_phantom() {
        let mut tracker = PhantomTracker::default();
        let phantoms = tracker.detect_gaps("Check the Redis cache", &[], 1);
        assert!(!phantoms.is_empty());
        let pid = phantoms[0].id;
        tracker.resolve(pid);
        assert!(tracker.get_active_phantoms().iter().all(|p| p.id != pid));
    }

    #[test]
    fn test_format_warnings() {
        let mut tracker = PhantomTracker::default();
        tracker.detect_gaps("Deploy to Kubernetes using Helm", &[], 1);
        let warnings = tracker.format_phantom_warnings();
        assert!(warnings.contains("WARNING"));
    }

    // --- EntityRegistry CRUD ---

    #[test]
    fn test_entity_registry_crud() {
        let mut reg = EntityRegistry::new();
        assert!(reg.is_empty());

        reg.register("Kubernetes");
        reg.register("Terraform");
        assert_eq!(reg.len(), 2);
        assert!(reg.is_known("Kubernetes"));
        assert!(!reg.is_known("Docker"));

        reg.unregister("Kubernetes");
        assert!(!reg.is_known("Kubernetes"));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_entity_registry_batch() {
        let mut reg = EntityRegistry::new();
        reg.register_batch(&["Redis", "Postgres", "Kafka"]);
        assert_eq!(reg.len(), 3);
        assert!(reg.is_known("Redis"));
        assert!(reg.is_known("Kafka"));
    }

    #[test]
    fn test_entity_registry_list() {
        let mut reg = EntityRegistry::new();
        reg.register_batch(&["B", "A", "C"]);
        let mut items = reg.list();
        items.sort();
        assert_eq!(items, vec!["A", "B", "C"]);
    }

    // --- detect_gaps_explicit ---

    #[test]
    fn test_detect_gaps_explicit_finds_unknown() {
        let mut tracker = PhantomTracker::default();
        tracker.register_entities(&["Kubernetes", "Terraform"]);

        let gaps = tracker.detect_gaps_explicit(
            "Deploy with Kubernetes and Docker",
            &["Kubernetes", "Docker"],
            1,
        );

        // Docker is not registered, so it should be flagged.
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].source_reference, "Docker");
    }

    #[test]
    fn test_detect_gaps_explicit_no_false_positives() {
        let mut tracker = PhantomTracker::default();
        tracker.register_entities(&["Kubernetes", "Terraform", "Helm"]);

        let gaps = tracker.detect_gaps_explicit(
            "Using Kubernetes and Helm",
            &["Kubernetes", "Helm"],
            1,
        );

        // Both are registered — no gaps.
        assert!(gaps.is_empty());
    }

    // --- Heuristic disabled via config ---

    #[test]
    fn test_heuristic_disabled() {
        let mut config = PhantomConfig::default();
        config.use_heuristic_detection = false;
        let mut tracker = PhantomTracker::new(config);

        // No entities registered and heuristic off — should find nothing.
        let gaps = tracker.detect_gaps(
            "Deploy to Kubernetes using Terraform",
            &[],
            1,
        );
        assert!(gaps.is_empty(), "Expected no gaps with heuristic disabled and empty registry, got: {:?}",
            gaps.iter().map(|g| &g.source_reference).collect::<Vec<_>>());
    }

    #[test]
    fn test_heuristic_disabled_with_registry() {
        let mut config = PhantomConfig::default();
        config.use_heuristic_detection = false;
        let mut tracker = PhantomTracker::new(config);
        tracker.register_entity("Kubernetes");

        // Registry-based detection should still work even with heuristic off.
        let gaps = tracker.detect_gaps(
            "Deploy to Kubernetes using Terraform",
            &[],
            1,
        );
        // Kubernetes is registered and referenced but not in known_entities → flagged.
        // Terraform is NOT registered and heuristic is off → not flagged.
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].source_reference, "Kubernetes");
    }

    // --- Integration: register, detect, resolve ---

    #[test]
    fn test_integration_register_detect_resolve() {
        let mut tracker = PhantomTracker::default();
        tracker.register_entities(&["Redis", "Postgres"]);

        // Caller mentions Redis, Kafka, Postgres — Kafka is unknown.
        let gaps = tracker.detect_gaps_explicit(
            "Need Redis and Kafka and Postgres",
            &["Redis", "Kafka", "Postgres"],
            1,
        );
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].source_reference, "Kafka");
        assert!(!gaps[0].resolved);

        // Resolve the gap.
        tracker.resolve(gaps[0].id);
        assert!(tracker.get_active_phantoms().is_empty());
    }

    // --- Heuristic fallback produces Low priority ---

    #[test]
    fn test_heuristic_fallback_low_priority() {
        let mut tracker = PhantomTracker::default();
        // No entities registered — heuristic only.
        // Use lowercase prefix so "Redis" is detected as a standalone capitalized word.
        let gaps = tracker.detect_gaps(
            "we use Redis for caching",
            &[],
            1,
        );
        let redis_gaps: Vec<_> = gaps.iter().filter(|g| g.source_reference == "Redis").collect();
        assert_eq!(redis_gaps.len(), 1);
        assert_eq!(redis_gaps[0].priority, PhantomPriority::Low);
    }

    #[test]
    fn test_registry_detection_higher_priority() {
        let mut tracker = PhantomTracker::default();
        tracker.register_entity("Redis");

        let gaps = tracker.detect_gaps("we use Redis for caching", &[], 1);
        let redis_gaps: Vec<_> = gaps.iter().filter(|g| g.source_reference == "Redis").collect();
        assert_eq!(redis_gaps.len(), 1);
        // Registry-based single-word entity → Medium priority.
        assert_eq!(redis_gaps[0].priority, PhantomPriority::Medium);
    }
}
