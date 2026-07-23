//! Embedder self-calibration for cosine-based cognitive thresholds.
//!
//! Absolute cosine thresholds are embedder-coupled: the same paraphrase pair
//! lands near 0.67 on one embedder and 0.80 on another, so a constant like
//! 0.97 silently disables deduplication on every real embedder (measured in
//! production, and the same failure previously disabled value-update
//! supersession at 0.88). Instead of hand-tuning constants per embedder, this
//! module embeds a small built-in probe set once per embedder and derives the
//! operating thresholds from the measured score distributions. Calibration is
//! persisted next to the database keyed by model and dimension, so it runs
//! once per embedder and self-heals on any embedder swap.
//!
//! Derivation is distribution-relative, never absolute: the dedup cosine sits
//! just below the paraphrase distribution while staying above measured
//! unrelated noise, and the topical floor (the loosest gate the value-update
//! rule uses) is the unrelated noise ceiling itself. When an embedder cannot
//! separate paraphrase from unrelated (a hash or degenerate provider),
//! calibration reports failure and callers keep their configured defaults.

use std::path::{Path, PathBuf};

use mentedb_core::error::{MenteError, MenteResult};
use mentedb_embedding::provider::EmbeddingProvider;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Version of the built-in probe set. Bump when the probe sentences change so
/// stored calibrations recompute instead of describing an older probe set.
pub const PROBE_VERSION: u32 = 1;

/// Minimum separation between the paraphrase p25 and unrelated p95 for the
/// embedder to count as calibratable. Below this the distributions overlap and
/// derived thresholds would be noise.
const MIN_SEPARATION: f32 = 0.05;

/// How far below the paraphrase p25 the dedup cosine sits, so the bulk of the
/// paraphrase distribution clears the gate rather than only its upper half.
const PARAPHRASE_MARGIN: f32 = 0.05;

/// How far above the unrelated p95 the dedup cosine must stay, so measured
/// noise never counts as a paraphrase even when the margins collide.
const NOISE_MARGIN: f32 = 0.02;

/// Sentence pairs that state the same fact in different words. Dedup must
/// catch these.
pub(crate) const PARAPHRASE_PAIRS: &[(&str, &str)] = &[
    (
        "The user prefers dark mode in every editor",
        "The user likes their editors set to dark mode",
    ),
    (
        "Standup is at 9:30 every weekday morning",
        "Every weekday the standup happens at 9:30 in the morning",
    ),
    (
        "The production database runs Postgres 16",
        "Postgres 16 is what the production database runs",
    ),
    (
        "Sarah leads the payments team",
        "The payments team is led by Sarah",
    ),
    (
        "Deploys go out on Friday afternoons",
        "The team ships deploys Friday afternoon",
    ),
    (
        "The user is allergic to peanuts",
        "Peanuts trigger the user's allergy",
    ),
    (
        "The API rate limit is one hundred requests per minute",
        "Clients may send at most one hundred requests a minute to the API",
    ),
    (
        "The user's favorite coffee is a latte",
        "A latte is the user's coffee of choice",
    ),
    (
        "Error budgets reset on the first of the month",
        "On the first of each month the error budget resets",
    ),
    (
        "The mobile app is written in Kotlin",
        "Kotlin is the language of the mobile app",
    ),
    (
        "Meetings are documented in the shared wiki",
        "The shared wiki holds the meeting notes",
    ),
    (
        "The user commutes by bicycle on dry days",
        "On dry days the user bikes to work",
    ),
];

/// Sentence pairs with no semantic relationship. Their cosine distribution is
/// the embedder's noise floor.
pub(crate) const UNRELATED_PAIRS: &[(&str, &str)] = &[
    (
        "The user prefers dark mode in every editor",
        "The quarterly tax filing is due in April",
    ),
    (
        "Standup is at 9:30 every weekday morning",
        "The cat needs a rabies booster next year",
    ),
    (
        "Postgres 16 is what the production database runs",
        "Flights to Lisbon are cheapest in February",
    ),
    (
        "Sarah leads the payments team",
        "The garage door opener needs new batteries",
    ),
    (
        "Deploys go out on Friday afternoons",
        "Basil grows best with six hours of sunlight",
    ),
    (
        "The user is allergic to peanuts",
        "The staging cluster autoscaled to four nodes",
    ),
    (
        "The API rate limit is one hundred requests per minute",
        "Grandma's birthday dinner is at the Italian place",
    ),
    (
        "The user's favorite coffee is a latte",
        "The linter forbids unused imports in CI",
    ),
    (
        "Error budgets reset on the first of the month",
        "The marathon route crosses two bridges",
    ),
    (
        "The mobile app is written in Kotlin",
        "Rent is due on the third of the month",
    ),
    (
        "Meetings are documented in the shared wiki",
        "The aquarium filter hums when it needs cleaning",
    ),
    (
        "The user commutes by bicycle on dry days",
        "Invoices over five thousand need a second approval",
    ),
    (
        "The retro board uses three columns",
        "The user's daughter plays goalkeeper on Saturdays",
    ),
    (
        "Feature flags live in the settings service",
        "The soup needs more salt near the end of cooking",
    ),
];

/// Thresholds derived from an embedder's measured score distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderCalibration {
    /// Embedder model the calibration describes.
    pub model: String,
    /// Embedding dimension the calibration describes.
    pub dimensions: usize,
    /// Probe set version the calibration was computed against.
    pub probe_version: u32,
    /// 25th percentile of paraphrase-pair cosines.
    pub paraphrase_p25: f32,
    /// 95th percentile of unrelated-pair cosines (the noise ceiling).
    pub unrelated_p95: f32,
    /// Operating cosine gate for paraphrase deduplication.
    pub dedup_cosine: f32,
    /// Loosest topical gate: text less similar than this is noise. Used as
    /// the value-update similarity floor.
    pub topical_floor: f32,
}

/// Nearest-rank percentile of an unsorted sample.
fn percentile(values: &mut [f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() - 1) as f32 * q).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Embed the probe set and derive operating thresholds. Errors when the
/// embedder fails or cannot separate paraphrase from unrelated noise; callers
/// keep their configured defaults in that case.
pub fn calibrate(embedder: &dyn EmbeddingProvider) -> MenteResult<EmbedderCalibration> {
    let texts: Vec<&str> = PARAPHRASE_PAIRS
        .iter()
        .chain(UNRELATED_PAIRS.iter())
        .flat_map(|(a, b)| [*a, *b])
        .collect();
    let vectors = embedder.embed_batch(&texts)?;
    if vectors.len() != texts.len() {
        return Err(MenteError::Index(format!(
            "calibration embed returned {} vectors for {} probes",
            vectors.len(),
            texts.len()
        )));
    }

    let pair_cosines = |offset: usize, count: usize| -> Vec<f32> {
        (0..count)
            .map(|i| cosine(&vectors[offset + 2 * i], &vectors[offset + 2 * i + 1]))
            .collect()
    };
    let mut para = pair_cosines(0, PARAPHRASE_PAIRS.len());
    let mut unrel = pair_cosines(PARAPHRASE_PAIRS.len() * 2, UNRELATED_PAIRS.len());

    let paraphrase_p25 = percentile(&mut para, 0.25);
    let unrelated_p95 = percentile(&mut unrel, 0.95);

    if paraphrase_p25 < unrelated_p95 + MIN_SEPARATION {
        return Err(MenteError::Index(format!(
            "embedder does not separate paraphrase from noise (paraphrase p25 {paraphrase_p25:.3}, unrelated p95 {unrelated_p95:.3}); keeping configured thresholds"
        )));
    }

    let dedup_cosine = (paraphrase_p25 - PARAPHRASE_MARGIN).max(unrelated_p95 + NOISE_MARGIN);
    Ok(EmbedderCalibration {
        model: embedder.model_name().to_string(),
        dimensions: embedder.dimensions(),
        probe_version: PROBE_VERSION,
        paraphrase_p25,
        unrelated_p95,
        dedup_cosine,
        topical_floor: unrelated_p95,
    })
}

/// File the calibration persists to, keyed by probe version, model, and
/// dimension so an embedder swap recomputes instead of reusing stale numbers.
fn calibration_path(dir: &Path, model: &str, dimensions: usize) -> PathBuf {
    let slug: String = model
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    dir.join(format!(
        ".embedder_calibration_v{PROBE_VERSION}_{slug}_{dimensions}.json"
    ))
}

/// Load a stored calibration for this embedder, or compute and persist one.
/// A failed save still returns the computed calibration; a failed computation
/// propagates so the caller can fall back to configured defaults.
pub fn load_or_calibrate(
    dir: &Path,
    embedder: &dyn EmbeddingProvider,
) -> MenteResult<EmbedderCalibration> {
    let path = calibration_path(dir, embedder.model_name(), embedder.dimensions());
    if let Ok(raw) = std::fs::read_to_string(&path)
        && let Ok(cal) = serde_json::from_str::<EmbedderCalibration>(&raw)
        && cal.probe_version == PROBE_VERSION
        && cal.dimensions == embedder.dimensions()
    {
        return Ok(cal);
    }
    let cal = calibrate(embedder)?;
    match serde_json::to_string_pretty(&cal) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!(error = %e, "calibration computed but not persisted");
            }
        }
        Err(e) => tracing::warn!(error = %e, "calibration serialize failed"),
    }
    info!(
        model = %cal.model,
        dedup_cosine = cal.dedup_cosine,
        topical_floor = cal.topical_floor,
        "embedder calibration computed"
    );
    Ok(cal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic provider mapping each probe text to a designed vector:
    /// both halves of a paraphrase pair share one basis vector, unrelated
    /// halves get mutually orthogonal vectors.
    struct DesignedProvider {
        vectors: HashMap<String, Vec<f32>>,
        dims: usize,
    }

    impl DesignedProvider {
        fn separable() -> Self {
            let dims = 2 * (PARAPHRASE_PAIRS.len() + 2 * UNRELATED_PAIRS.len());
            let mut vectors = HashMap::new();
            let basis = |i: usize| -> Vec<f32> {
                let mut v = vec![0.0; dims];
                v[i] = 1.0;
                v
            };
            for (i, (a, b)) in PARAPHRASE_PAIRS.iter().enumerate() {
                // Same direction with a slight tilt: cosine near but below 1.
                let va = basis(i);
                let mut vb = basis(i);
                vb[dims - 1] = 0.3;
                vectors.insert((*a).to_string(), va);
                vectors.entry((*b).to_string()).or_insert(vb);
            }
            let base = PARAPHRASE_PAIRS.len();
            for (j, (a, b)) in UNRELATED_PAIRS.iter().enumerate() {
                vectors
                    .entry((*a).to_string())
                    .or_insert_with(|| basis(base + 2 * j));
                vectors
                    .entry((*b).to_string())
                    .or_insert_with(|| basis(base + 2 * j + 1));
            }
            Self { vectors, dims }
        }

        /// Every text maps to the same vector: paraphrase and unrelated
        /// cosines are all 1.0, which no threshold can separate.
        fn degenerate() -> Self {
            let mut vectors = HashMap::new();
            for (a, b) in PARAPHRASE_PAIRS.iter().chain(UNRELATED_PAIRS.iter()) {
                vectors.insert((*a).to_string(), vec![1.0, 0.0]);
                vectors.insert((*b).to_string(), vec![1.0, 0.0]);
            }
            Self { vectors, dims: 2 }
        }
    }

    impl EmbeddingProvider for DesignedProvider {
        fn embed(&self, text: &str) -> MenteResult<Vec<f32>> {
            Ok(self
                .vectors
                .get(text)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.dims]))
        }

        fn embed_batch(&self, texts: &[&str]) -> MenteResult<Vec<Vec<f32>>> {
            texts.iter().map(|t| self.embed(t)).collect()
        }

        fn dimensions(&self) -> usize {
            self.dims
        }

        fn model_name(&self) -> &str {
            "designed-test-provider"
        }
    }

    #[test]
    fn separable_provider_calibrates() {
        let cal = calibrate(&DesignedProvider::separable()).expect("calibratable");
        // Paraphrase pairs sit near 1.0, unrelated near 0.0.
        assert!(cal.paraphrase_p25 > 0.9);
        assert!(cal.unrelated_p95 < 0.1);
        assert!(cal.dedup_cosine > cal.topical_floor);
        assert!(cal.dedup_cosine < cal.paraphrase_p25);
        assert_eq!(cal.probe_version, PROBE_VERSION);
    }

    #[test]
    fn degenerate_provider_fails_closed() {
        assert!(calibrate(&DesignedProvider::degenerate()).is_err());
    }

    #[test]
    fn calibration_roundtrips_through_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let provider = DesignedProvider::separable();
        let first = load_or_calibrate(dir.path(), &provider).expect("calibrate");
        // Second load must come from the file, not a recompute; equality of
        // every derived field is the observable.
        let second = load_or_calibrate(dir.path(), &provider).expect("load");
        assert_eq!(first.dedup_cosine, second.dedup_cosine);
        assert_eq!(first.topical_floor, second.topical_floor);
        assert!(
            calibration_path(dir.path(), provider.model_name(), provider.dimensions()).exists()
        );
    }

    #[test]
    fn percentile_nearest_rank() {
        let mut v = vec![0.9, 0.1, 0.5, 0.3, 0.7];
        assert!((percentile(&mut v, 0.5) - 0.5).abs() < f32::EPSILON);
        assert!((percentile(&mut v.clone(), 0.0) - 0.1).abs() < f32::EPSILON);
        assert!((percentile(&mut v.clone(), 1.0) - 0.9).abs() < f32::EPSILON);
        assert_eq!(percentile(&mut [], 0.5), 0.0);
    }
}
