use mentedb_core::types::{MemoryId, Timestamp};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PainSignal {
    pub id: Uuid,
    pub memory_id: MemoryId,
    pub intensity: f32,
    pub trigger_keywords: Vec<String>,
    pub description: String,
    pub created_at: Timestamp,
    pub decay_rate: f32,
}

pub struct PainRegistry {
    signals: Vec<PainSignal>,
}

impl PainRegistry {
    pub fn new() -> Self {
        Self {
            signals: Vec::new(),
        }
    }

    pub fn record_pain(&mut self, signal: PainSignal) {
        self.signals.push(signal);
    }

    pub fn get_pain_for_context(&self, context_keywords: &[String]) -> Vec<&PainSignal> {
        let ctx_lower: Vec<String> = context_keywords.iter().map(|k| k.to_lowercase()).collect();

        let mut scored: Vec<(&PainSignal, f32)> = self
            .signals
            .iter()
            .filter_map(|signal| {
                let matched = signal
                    .trigger_keywords
                    .iter()
                    .filter(|trigger| ctx_lower.iter().any(|ctx| ctx.contains(&trigger.to_lowercase())))
                    .count();

                if matched > 0 {
                    let relevance = matched as f32 / signal.trigger_keywords.len().max(1) as f32;
                    let score = signal.intensity * relevance;
                    Some((signal, score))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(s, _)| s).collect()
    }

    pub fn decay_all(&mut self, current_time: Timestamp) {
        for signal in &mut self.signals {
            signal.intensity = Self::compute_decayed(signal, current_time);
        }
    }

    pub fn current_intensity(signal: &PainSignal, now: Timestamp) -> f32 {
        Self::compute_decayed(signal, now)
    }

    fn compute_decayed(signal: &PainSignal, now: Timestamp) -> f32 {
        if now <= signal.created_at {
            return signal.intensity;
        }
        let time_delta = (now - signal.created_at) as f64;
        let decayed = signal.intensity as f64 * (-signal.decay_rate as f64 * time_delta).exp();
        decayed as f32
    }

    pub fn format_pain_warnings(signals: &[&PainSignal]) -> String {
        if signals.is_empty() {
            return String::new();
        }

        signals
            .iter()
            .take(5)
            .map(|s| {
                format!(
                    "CAUTION: {} (pain: {:.2}). Triggers: [{}]",
                    s.description,
                    s.intensity,
                    s.trigger_keywords.join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for PainRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(keywords: Vec<&str>, intensity: f32) -> PainSignal {
        PainSignal {
            id: Uuid::new_v4(),
            memory_id: Uuid::new_v4(),
            intensity,
            trigger_keywords: keywords.into_iter().map(String::from).collect(),
            description: "Test pain signal".to_string(),
            created_at: 1000,
            decay_rate: 0.0001,
        }
    }

    #[test]
    fn test_pain_trigger_matching() {
        let mut registry = PainRegistry::new();
        registry.record_pain(make_signal(vec!["mongodb", "nosql"], 0.95));
        registry.record_pain(make_signal(vec!["python", "flask"], 0.5));

        let results = registry.get_pain_for_context(&["mongodb".to_string(), "database".to_string()]);
        assert_eq!(results.len(), 1);
        assert!(results[0].trigger_keywords.contains(&"mongodb".to_string()));
    }

    #[test]
    fn test_decay() {
        let signal = make_signal(vec!["test"], 1.0);
        let decayed = PainRegistry::current_intensity(&signal, 11000);
        assert!(decayed < 1.0, "Expected decay, got {}", decayed);
        assert!(decayed > 0.0, "Expected positive intensity, got {}", decayed);
    }

    #[test]
    fn test_format_warnings() {
        let s = make_signal(vec!["mongodb", "nosql"], 0.95);
        let warnings = PainRegistry::format_pain_warnings(&[&s]);
        assert!(warnings.contains("CAUTION"));
        assert!(warnings.contains("0.95"));
    }
}
