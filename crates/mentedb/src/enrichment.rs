//! Sleeptime enrichment pipeline orchestrator.
//!
//! Runs the full 4-phase enrichment pipeline:
//! 1. **Batch LLM Extraction** — episodic memories → semantic + entities
//! 2. **Entity Linking** — rule-based + LLM resolution
//! 3. **Community Detection** — category clustering + LLM summaries
//! 4. **User Model** — always-scoped profile from all knowledge
//!
//! This module requires the `enrichment` feature flag.

use std::collections::HashSet;

use mentedb_cognitive::llm::{CognitiveLlmService, LlmJudge};
use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::AgentId;
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_extraction::{ExtractionConfig, ExtractionPipeline, HttpExtractionProvider};

use crate::MenteDb;

/// Result of a full enrichment run.
#[derive(Debug, Default)]
pub struct EnrichmentResult {
    /// Semantic memories stored from extraction.
    pub memories_stored: usize,
    /// Graph edges created across all phases.
    pub edges_created: usize,
    /// Entity nodes extracted.
    pub entities_extracted: usize,
    /// Duplicate memories skipped.
    pub duplicates_skipped: usize,
    /// Contradictions detected.
    pub contradictions_found: usize,
    /// Entities linked via sync (cache) resolution.
    pub sync_linked: usize,
    /// Entities linked via LLM resolution.
    pub llm_linked: usize,
    /// Community summaries created.
    pub communities_created: usize,
    /// Whether the user profile was updated.
    pub user_model_updated: bool,
}

/// Run the full 4-phase enrichment pipeline.
///
/// This is the main entry point for enrichment. Call it when
/// `db.needs_enrichment()` returns true.
///
/// # Arguments
/// * `db` — The MenteDb instance
/// * `extraction_config` — LLM extraction provider config
/// * `embedder` — Embedding provider for new memories
/// * `cognitive_llm` — Optional LLM service for entity linking, community detection, and user model
/// * `current_turn` — Current conversation turn (for marking enrichment complete)
pub async fn run_enrichment<J: LlmJudge>(
    db: &MenteDb,
    extraction_config: ExtractionConfig,
    embedder: &dyn EmbeddingProvider,
    cognitive_llm: Option<&CognitiveLlmService<J>>,
    current_turn: u64,
) -> EnrichmentResult {
    let mut result = EnrichmentResult::default();

    let candidates = db.enrichment_candidates();
    if candidates.is_empty() {
        tracing::debug!("enrichment: no candidates, marking complete");
        db.mark_enrichment_complete(current_turn);
        return result;
    }

    tracing::info!(
        candidates = candidates.len(),
        "starting sleeptime enrichment"
    );

    // ── Phase 1: Batch LLM Extraction ──
    let http_provider = match HttpExtractionProvider::new(extraction_config.clone()) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = %e, "enrichment: failed to create HTTP provider");
            db.mark_enrichment_complete(current_turn);
            return result;
        }
    };

    let enrichment_config = db.enrichment_config().clone();
    let pipeline_cfg = ExtractionConfig {
        quality_threshold: enrichment_config.min_confidence,
        ..extraction_config
    };
    let pipeline = ExtractionPipeline::new(http_provider, pipeline_cfg);

    let batches = batch_conversations(&candidates, 10);

    for (batch_idx, batch) in batches.iter().enumerate() {
        let conversation = batch
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");

        let source_ids: Vec<mentedb_core::types::MemoryId> = batch.iter().map(|m| m.id).collect();

        // Get existing memories for dedup
        let existing: Vec<MemoryNode> =
            if let Ok(Some(emb)) = db.embed_text(&conversation[..conversation.len().min(500)]) {
                db.recall_similar(&emb, 30)
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|(id, _)| db.get_memory(id).ok())
                    .collect()
            } else {
                Vec::new()
            };

        match pipeline.process(&conversation, &existing, embedder).await {
            Ok(extraction_result) => {
                result.duplicates_skipped += extraction_result.stats.rejected_duplicate;
                result.contradictions_found += extraction_result.stats.contradictions_found;

                let mut nodes = Vec::new();
                for mem in extraction_result
                    .to_store
                    .iter()
                    .chain(extraction_result.contradictions.iter().map(|(m, _)| m))
                {
                    let mem_type =
                        mentedb_extraction::map_extraction_type_to_memory_type(&mem.memory_type);
                    let embedding = match embedder.embed(&mem.content) {
                        Ok(e) => e,
                        Err(e) => {
                            tracing::warn!(error = %e, "enrichment: embedding failed");
                            continue;
                        }
                    };
                    let mut node =
                        MemoryNode::new(AgentId::nil(), mem_type, mem.content.clone(), embedding);
                    node.tags = mem.tags.clone();
                    node.confidence = mem.confidence;
                    nodes.push(node);
                }

                // Build entity nodes
                for entity in &extraction_result.entities {
                    let content = entity.to_content();
                    let embedding_text = entity.embedding_key();
                    let embedding = match embedder.embed(&embedding_text) {
                        Ok(e) => e,
                        Err(e) => {
                            tracing::warn!(error = %e, "enrichment: entity embedding failed");
                            continue;
                        }
                    };
                    let mut node =
                        MemoryNode::new(AgentId::nil(), MemoryType::Semantic, content, embedding);
                    let name_lower = entity.name.to_lowercase();
                    node.tags.push(format!("entity:{}", name_lower));
                    node.tags
                        .push(format!("entity_type:{}", entity.entity_type));
                    if let Some(cat) = entity.attributes.get("category") {
                        for c in cat.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                            node.tags.push(format!("category:{}", c.to_lowercase()));
                        }
                    }
                    nodes.push(node);
                    result.entities_extracted += 1;
                }

                match db.store_enrichment_memories(nodes, &source_ids) {
                    Ok((stored, edges)) => {
                        result.memories_stored += stored;
                        result.edges_created += edges;
                    }
                    Err(e) => {
                        tracing::error!(batch = batch_idx, error = %e, "enrichment: failed to store batch");
                    }
                }
            }
            Err(e) => {
                tracing::error!(batch = batch_idx, error = %e, "enrichment: extraction failed");
            }
        }
    }

    // ── Phase 2: Entity Linking ──
    let sync_link_result = match db.link_entities() {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "enrichment: sync entity linking failed");
            crate::EntityLinkResult::default()
        }
    };
    result.sync_linked = sync_link_result.linked;
    result.edges_created += sync_link_result.edges_created;

    if let Some(llm) = cognitive_llm {
        let llm_link = run_llm_entity_linking(db, llm).await;
        result.llm_linked = llm_link.linked;
        result.edges_created += llm_link.edges_created;
    }

    // ── Phase 3: Community Detection ──
    if let Some(llm) = cognitive_llm {
        result.communities_created = run_community_detection(db, llm).await;
    }

    // ── Phase 4: User Model ──
    if let Some(llm) = cognitive_llm {
        result.user_model_updated = run_user_model(db, llm).await;
    }

    db.mark_enrichment_complete(current_turn);

    tracing::info!(
        stored = result.memories_stored,
        edges = result.edges_created,
        entities = result.entities_extracted,
        sync_linked = result.sync_linked,
        llm_linked = result.llm_linked,
        duplicates_skipped = result.duplicates_skipped,
        contradictions = result.contradictions_found,
        communities = result.communities_created,
        user_model = result.user_model_updated,
        batches = batches.len(),
        "sleeptime enrichment complete"
    );

    result
}

/// Phase 2b: LLM entity resolution for unresolved entities.
async fn run_llm_entity_linking<J: LlmJudge>(
    db: &MenteDb,
    llm: &CognitiveLlmService<J>,
) -> crate::EntityLinkResult {
    let all_entities_with_context = db.entity_names_with_context();
    let unresolved = db.unresolved_entity_names();

    if unresolved.is_empty() {
        return crate::EntityLinkResult::default();
    }

    tracing::info!(
        total_entities = all_entities_with_context.len(),
        unresolved = unresolved.len(),
        "running LLM entity resolution"
    );

    let mut candidates: Vec<mentedb_cognitive::EntityCandidate> = all_entities_with_context
        .iter()
        .map(|(name, ctx)| mentedb_cognitive::EntityCandidate {
            name: name.clone(),
            context: ctx.clone(),
            memory_id: None,
        })
        .collect();
    candidates.sort_by(|a, b| a.name.cmp(&b.name));

    let batch_size = 50;
    let mut all_merge_groups = Vec::new();

    for chunk in candidates.chunks(batch_size) {
        match llm.resolve_entities(chunk).await {
            Ok(groups) => all_merge_groups.extend(groups),
            Err(e) => tracing::error!(error = %e, "enrichment: LLM entity resolution failed"),
        }
    }

    if all_merge_groups.len() > 1 {
        all_merge_groups = consolidate_merge_groups(all_merge_groups);
    }

    if all_merge_groups.is_empty() {
        return crate::EntityLinkResult::default();
    }

    let resolutions: Vec<crate::EntityLinkResolution> = all_merge_groups
        .iter()
        .map(|g| crate::EntityLinkResolution {
            canonical: g.canonical.clone(),
            aliases: g.aliases.clone(),
            confidence: g.confidence,
        })
        .collect();

    let separations: Vec<crate::EntitySeparation> = Vec::new();

    match db.apply_entity_link_resolutions(&resolutions, &separations) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = %e, "enrichment: failed to apply entity resolutions");
            crate::EntityLinkResult::default()
        }
    }
}

/// Phase 3: Community detection — group entities by category, generate LLM summaries.
async fn run_community_detection<J: LlmJudge>(db: &MenteDb, llm: &CognitiveLlmService<J>) -> usize {
    let communities = db.entity_communities();
    let existing_summaries: HashSet<String> = db
        .community_summaries()
        .iter()
        .flat_map(|m| {
            m.tags
                .iter()
                .filter_map(|t| t.strip_prefix("community:").map(|s| s.to_string()))
        })
        .collect();

    let mut created = 0;

    for (category, members) in &communities {
        if existing_summaries.contains(category) {
            tracing::debug!(category = %category, "community summary already exists, skipping");
            continue;
        }

        let entity_pairs: Vec<(String, String)> = members.iter().take(20).cloned().collect();

        match llm
            .generate_community_summary(category, &entity_pairs)
            .await
        {
            Ok(summary) => {
                let member_names: Vec<String> =
                    entity_pairs.iter().map(|(n, _)| n.clone()).collect();
                match db.store_community_summary(category, &summary.summary, &member_names) {
                    Ok(_) => {
                        created += 1;
                        tracing::info!(category = %category, members = members.len(), "community summary created");
                    }
                    Err(e) => {
                        tracing::error!(error = %e, category = %category, "failed to store community summary");
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, category = %category, "LLM community summary failed");
            }
        }
    }

    created
}

/// Phase 4: User model generation — synthesize always-scoped profile.
async fn run_user_model<J: LlmJudge>(db: &MenteDb, llm: &CognitiveLlmService<J>) -> bool {
    let facts = db.profile_facts();
    let community_texts: Vec<String> = db
        .community_summaries()
        .iter()
        .map(|m| m.content.clone())
        .collect();

    if facts.len() < 3 && community_texts.is_empty() {
        return false;
    }

    match llm.generate_user_profile(&facts, &community_texts).await {
        Ok(profile) => match db.store_user_profile(&profile.profile) {
            Ok(_) => {
                tracing::info!("user profile updated");
                true
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to store user profile");
                false
            }
        },
        Err(e) => {
            tracing::warn!(error = %e, "LLM user profile generation failed");
            false
        }
    }
}

/// Split memories into batches.
fn batch_conversations(memories: &[MemoryNode], batch_size: usize) -> Vec<Vec<MemoryNode>> {
    memories
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Transitively consolidate merge groups from independent LLM batches.
///
/// Uses union-find to merge groups that share entity names across batch boundaries.
fn consolidate_merge_groups(
    groups: Vec<mentedb_cognitive::llm::EntityMergeGroup>,
) -> Vec<mentedb_cognitive::llm::EntityMergeGroup> {
    use std::collections::{HashMap, HashSet};

    let mut name_to_group: HashMap<String, usize> = HashMap::new();
    let mut parent: Vec<usize> = (0..groups.len()).collect();

    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    for (idx, group) in groups.iter().enumerate() {
        let all_names = std::iter::once(&group.canonical).chain(group.aliases.iter());
        for name in all_names {
            let key = name.to_lowercase();
            if let Some(&existing_idx) = name_to_group.get(&key) {
                union(&mut parent, idx, existing_idx);
            }
            name_to_group.insert(key, idx);
        }
    }

    let mut root_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..groups.len() {
        let root = find(&mut parent, i);
        root_groups.entry(root).or_default().push(i);
    }

    root_groups
        .into_values()
        .map(|indices| {
            let mut all_names: HashSet<String> = HashSet::new();
            let mut best_confidence: f32 = 0.0;
            let mut canonical = String::new();

            for &idx in &indices {
                let g = &groups[idx];
                all_names.insert(g.canonical.clone());
                for a in &g.aliases {
                    all_names.insert(a.clone());
                }
                if g.confidence > best_confidence {
                    best_confidence = g.confidence;
                    canonical = g.canonical.clone();
                }
            }

            all_names.remove(&canonical);
            mentedb_cognitive::llm::EntityMergeGroup {
                canonical,
                aliases: all_names.into_iter().collect(),
                confidence: best_confidence,
            }
        })
        .collect()
}
