//! 存储模式切换、冷向量配置及固定候选精排。

use super::*;
use crate::{RerankConfig, StorageMode, VectorStorageStats};

impl Index {
    pub fn open_ssd(path: impl AsRef<Path>, config: crate::SsdCacheConfig) -> Result<Self> {
        let mut index = Self::load_mmap(path)?;
        index.set_ssd_cache_config(config)?;
        Ok(index)
    }

    /// 为已映射向量启用直接读取和共享缓存，成功后替换原有运行时及其缓冲池。
    pub fn set_ssd_cache_config(&mut self, config: crate::SsdCacheConfig) -> Result<()> {
        config.validate()?;
        if self.storage_mode != StorageMode::Mmap {
            return Err(Error::InvalidRerankConfig(
                "请先打开映射索引，再配置直接 SSD 缓存",
            ));
        }
        if let Some(store) = &mut self.rerank {
            store.enable_direct(config)?;
        } else {
            let mut store = ColdStore::canonical(
                self.dimensions,
                &self.vectors,
                self.metric,
                &self.boundary_factors,
            )?;
            store.enable_direct(config)?;
            self.rerank = Some(store);
        }
        Ok(())
    }

    pub fn ssd_cache_config(&self) -> Option<crate::SsdCacheConfig> {
        self.rerank
            .as_ref()
            .and_then(|store| store.direct.as_ref())
            .map(|reader| reader.config)
    }

    pub fn ssd_io_stats(&self) -> Option<crate::SsdIoStats> {
        self.rerank
            .as_ref()
            .and_then(|store| store.direct.as_ref())
            .map(|reader| reader.stats())
    }

    /// 等待已提交预读结束后清空共享缓存，调用期间由可变借用排除查询。
    pub fn clear_ssd_cache(&mut self) {
        if let Some(reader) = self.rerank.as_ref().and_then(|store| store.direct.as_ref()) {
            reader.clear();
        }
    }

    /// 在查询调用结束后等待已提交预读排空，便于完整记录后台 I/O。
    pub fn drain_ssd_reads(&mut self) {
        if let Some(reader) = self.rerank.as_ref().and_then(|store| store.direct.as_ref()) {
            reader.wait_idle();
        }
    }

    pub fn rerank_config(&self) -> RerankConfig {
        self.rerank
            .as_ref()
            .map_or(RerankConfig::default(), |store| store.config)
    }

    /// 先完整验证新编码，再原子替换当前重排表示，图拓扑和原始向量保持同一代。
    pub fn set_rerank_config(&mut self, config: RerankConfig) -> Result<()> {
        config.validate()?;
        let ssd_config = self.ssd_cache_config();
        let mut store = if config == RerankConfig::default() && ssd_config.is_none() {
            None
        } else if config == RerankConfig::default() {
            Some(ColdStore::canonical(
                self.dimensions,
                &self.vectors,
                self.metric,
                &self.boundary_factors,
            )?)
        } else {
            Some(ColdStore::build(
                config,
                self.dimensions,
                &self.vectors,
                &self.active,
                self.metric,
            )?)
        };
        if self.storage_mode == StorageMode::Mmap
            && let Some(cold) = &mut store
        {
            cold.bytes = cold.bytes.file_backed()?;
        }
        if let Some(ssd_config) = ssd_config
            && let Some(cold) = &mut store
        {
            cold.enable_direct(ssd_config)?;
        }
        // 编码准备完成后，原始向量只保留文件后备；查询主存仅持有当前重排表示。
        let vectors = match (self.storage_mode, store.is_some()) {
            (StorageMode::Memory, false) => self.vectors.to_owned(),
            _ => self.vectors.file_backed()?,
        };
        self.vectors = vectors;
        self.rerank = store;
        Ok(())
    }

    pub fn storage_mode(&self) -> StorageMode {
        self.storage_mode
    }

    /// 切换运行时存储位置；临时映射由索引持有，持久化通过 save 发布。
    pub fn set_storage_mode(&mut self, mode: StorageMode) -> Result<()> {
        if mode == self.storage_mode {
            return Ok(());
        }
        let vectors = match mode {
            StorageMode::Memory if self.rerank.is_none() => self.vectors.to_owned(),
            _ => self.vectors.file_backed()?,
        };
        let cold_bytes = self
            .rerank
            .as_ref()
            .map(|store| match mode {
                StorageMode::Memory => Ok(store.bytes.to_owned()),
                StorageMode::Mmap => store.bytes.file_backed(),
            })
            .transpose()?;
        self.vectors = vectors;
        if let Some(store) = &mut self.rerank {
            store.bytes = cold_bytes.unwrap();
            store.direct = None;
        }
        self.storage_mode = mode;
        Ok(())
    }

    pub fn vector_storage_stats(&self) -> VectorStorageStats {
        VectorStorageStats {
            canonical_heap_bytes: self.vectors.heap_bytes(),
            canonical_mapped_bytes: self.vectors.mapped_bytes(),
            rerank_heap_bytes: self
                .rerank
                .as_ref()
                .map_or(0, |store| store.bytes.heap_bytes()),
            rerank_mapped_bytes: self
                .rerank
                .as_ref()
                .map_or(0, |store| store.bytes.mapped_bytes()),
            shared_metadata_bytes: self
                .rerank
                .as_ref()
                .map_or(0, |store| store.center.len() * 4),
            rerank_metadata_bytes: self
                .rerank
                .as_ref()
                .map_or(0, |store| std::mem::size_of_val(store.prepared.as_slice())),
            rerank_record_bytes: self
                .rerank
                .as_ref()
                .map_or(self.dimensions * 4, |store| store.stride),
        }
    }

    pub(crate) fn native_space(&self, curvature: f64, rerank: bool) -> NativeSpace<'_> {
        let (vectors, delta) = self.vectors.slices();
        NativeSpace {
            vectors,
            delta,
            rerank: if rerank { self.rerank.as_ref() } else { None },
            boundary_factors: &self.boundary_factors,
            dimensions: self.dimensions,
            curvature,
        }
    }

    /// 对给定标识执行当前冷向量表示的纯重排，跳过不存在、已删除及重复标识。
    pub fn rerank_candidates(
        &self,
        query: &[f32],
        candidates: &[u64],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>> {
        if top_k == 0 {
            return Err(Error::InvalidSearchConfig("top_k 必须大于零"));
        }
        validate_vector(query, self.dimensions)?;
        let factor = self.metric.boundary_factor(query, 0)?;
        let mut seen = HashSet::new();
        let mut buffer = crate::storage::RerankScratch::default();
        let mut scores = Vec::with_capacity(candidates.len().min(self.len()));
        for &id in candidates {
            if !seen.insert(id) {
                continue;
            }
            let Some(&slot) = self.id_to_slot.get(&id) else {
                continue;
            };
            let score = if let Some(store) = &self.rerank {
                let value = store.score(query, factor, slot, self.metric, &mut buffer)?;
                match self.metric {
                    Metric::Cosine => value,
                    Metric::Poincare { curvature } => -poincare_distance(value, curvature),
                }
            } else {
                let vector = self.vectors.row(slot, self.dimensions);
                match self.metric {
                    Metric::Cosine => f64::from(cosine_similarity_f32(query, vector)),
                    Metric::Poincare { curvature } => -poincare_distance(
                        poincare_proxy(query, vector, factor * self.boundary_factors[slot]),
                        curvature,
                    ),
                }
            };
            scores.push((id, score));
        }
        rank_scores(scores, top_k)
    }
}

pub(super) fn rank_scores(mut scores: Vec<(u64, f64)>, top_k: usize) -> Result<Vec<(u64, f32)>> {
    let order = |a: &(u64, f64), b: &(u64, f64)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    if scores.len() > top_k {
        scores.select_nth_unstable_by(top_k, order);
        scores.truncate(top_k);
    }
    scores.sort_unstable_by(order);
    if scores
        .iter()
        .any(|(_, score)| !score.is_finite() || !(*score as f32).is_finite())
    {
        return Err(Error::DistanceOverflow);
    }
    Ok(scores
        .into_iter()
        .map(|(id, score)| (id, score as f32))
        .collect())
}
