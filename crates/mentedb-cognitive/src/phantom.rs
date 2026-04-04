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

pub struct PhantomTracker {
    phantoms: Vec<PhantomMemory>,
}

impl PhantomTracker {
    pub fn new() -> Self {
        Self {
            phantoms: Vec::new(),
        }
    }

    /// Scan content for entity references not in known_entities.
    /// Detects: capitalized multi-word terms, quoted terms, technical terms (containing
    /// hyphens/dots/underscores or ALL_CAPS).
    pub fn detect_gaps(
        &mut self,
        content: &str,
        known_entities: &[String],
        turn_id: u64,
    ) -> Vec<PhantomMemory> {
        let known_lower: Vec<String> = known_entities.iter().map(|e| e.to_lowercase()).collect();
        let mut detected = Vec::new();
        let mut seen = ahash::AHashSet::new();

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
                // Collect consecutive capitalized words
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
                // Skip common English words that happen to be capitalized (start of sentence)
                let common = [
                    "the", "a", "an", "is", "are", "was", "were", "it", "this", "that",
                    "we", "you", "they", "he", "she", "i", "my", "our", "but", "and",
                    "or", "if", "then", "when", "how", "what", "where", "who", "do",
                    "does", "did", "have", "has", "had", "be", "been", "being", "so",
                    "also", "just", "very", "too", "not", "no", "yes", "can", "will",
                    "should", "would", "could", "may", "might", "must", "shall",
                    "note", "see", "use", "make", "let", "new", "set", "get",
                ];
                if entity.split_whitespace().count() == 1 && common.contains(&entity.to_lowercase().as_str()) {
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
                // Check for technical terms: contains hyphens/dots/underscores or ALL_CAPS
                if !w.is_empty() && w.len() >= 3 {
                    let is_technical = w.contains('-') || w.contains('.') || w.contains('_')
                        || (w.len() >= 3 && w.chars().all(|c| c.is_uppercase() || c.is_ascii_digit() || c == '_'));

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

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let new_phantoms: Vec<PhantomMemory> = detected
            .into_iter()
            .map(|entity| {
                let priority = if entity.chars().all(|c| c.is_uppercase() || c.is_ascii_digit() || c == '_' || c == ' ') {
                    PhantomPriority::High
                } else if entity.split_whitespace().count() > 1 {
                    PhantomPriority::Medium
                } else {
                    PhantomPriority::Low
                };

                PhantomMemory {
                    id: Uuid::new_v4(),
                    gap_description: format!("No stored knowledge about '{}'", entity),
                    source_reference: entity,
                    source_turn: turn_id,
                    priority,
                    created_at: now,
                    resolved: false,
                }
            })
            .collect();

        self.phantoms.extend(new_phantoms.clone());
        new_phantoms
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
            .take(5)
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
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_unknown_entities() {
        let mut tracker = PhantomTracker::new();
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
        let mut tracker = PhantomTracker::new();
        let phantoms = tracker.detect_gaps("Check the Redis cache", &[], 1);
        assert!(!phantoms.is_empty());
        let pid = phantoms[0].id;
        tracker.resolve(pid);
        assert!(tracker.get_active_phantoms().iter().all(|p| p.id != pid));
    }

    #[test]
    fn test_format_warnings() {
        let mut tracker = PhantomTracker::new();
        tracker.detect_gaps("Deploy to Kubernetes using Helm", &[], 1);
        let warnings = tracker.format_phantom_warnings();
        assert!(warnings.contains("WARNING"));
    }
}
