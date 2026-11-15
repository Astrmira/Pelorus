//! 公开查询入口与精确距离计算。

use super::*;
impl Index {
    pub fn search(&self, query: &[f32], config: &SearchConfig) -> Result<Vec<(u64, f32)>> {
        self.search_impl(query, config, self.navigation)
    }

    pub fn poincare_navigation(&self) -> PoincareNavigation {
        self.navigation
    }

    /// 在同一张图上以精确距离导航，用于隔离量化导航带来的成本与召回变化。
    #[cfg(feature = "research")]
    pub fn search_poincare_exact_navigation(
        &self,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<Vec<(u64, f32)>> {
        self.search_impl(query, config, PoincareNavigation::Exact)
    }

    /// 在同一张图上以 SM-Poincaré 2-bit 量化代理导航并有界精排（支持 Exact 构建图的跨模式测试）。
    #[cfg(feature = "research")]
    pub fn search_poincare_quantized_navigation(
        &self,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<Vec<(u64, f32)>> {
        self.search_impl(query, config, PoincareNavigation::GeometryBq2)
    }

    /// 遵循 ICDE QuIVer Eq. (60) 执行三阶段精确候选损失分解诊断。
    #[cfg(feature = "research")]
    pub fn search_loss_decomposition(
        &self,
        query: &[f32],
        config: &SearchConfig,
        navigation: PoincareNavigation,
        truth_ids: &[u64],
    ) -> Result<crate::graph::SearchLossStats> {
        validate_search_config(config)?;
        validate_vector(query, self.dimensions)?;
        let factor = self.metric.boundary_factor(query, 0)?;
        if self.is_empty() {
            return Ok(crate::graph::SearchLossStats::default());
        }
        match self.metric {
            Metric::Cosine => Err(Error::InvalidSearchConfig("损失分解当前专用于双曲流形诊断")),
            Metric::Poincare { curvature } => Ok(self.graph.search_native_loss_stats(
                &NativeSpace {
                    vectors: &self.vectors,
                    boundary_factors: &self.boundary_factors,
                    dimensions: self.dimensions,
                    curvature,
                },
                query,
                factor,
                config,
                navigation == PoincareNavigation::GeometryBq2,
                truth_ids,
            )),
        }
    }

    /// 零 HashSet 插桩污染采集单次双曲查询的四阶段纯墙钟耗时分解（路由选种 / 主导航束 / 非对称过滤器 / 精确精排）。
    #[cfg(feature = "research")]
    pub fn search_stage_profile(
        &self,
        query: &[f32],
        config: &SearchConfig,
        navigation: PoincareNavigation,
    ) -> Result<crate::graph::SearchStageProfile> {
        validate_search_config(config)?;
        validate_vector(query, self.dimensions)?;
        let factor = self.metric.boundary_factor(query, 0)?;
        if self.is_empty() {
            return Ok(crate::graph::SearchStageProfile::default());
        }
        match self.metric {
            Metric::Cosine => Err(Error::InvalidSearchConfig("阶段剖析当前专用于双曲流形诊断")),
            Metric::Poincare { curvature } => Ok(self.graph.search_native_stage_profile(
                &NativeSpace {
                    vectors: &self.vectors,
                    boundary_factors: &self.boundary_factors,
                    dimensions: self.dimensions,
                    curvature,
                },
                query,
                factor,
                config,
                navigation == PoincareNavigation::GeometryBq2,
            )),
        }
    }

    pub(crate) fn search_impl(
        &self,
        query: &[f32],
        config: &SearchConfig,
        navigation: PoincareNavigation,
    ) -> Result<Vec<(u64, f32)>> {
        validate_search_config(config)?;
        validate_vector(query, self.dimensions)?;
        let factor = self.metric.boundary_factor(query, 0)?;
        if self.is_empty() {
            return Ok(Vec::new());
        }
        match self.metric {
            Metric::Cosine => Ok(self.graph.search_flat(query, &self.vectors, config)),
            Metric::Poincare { curvature } => self
                .graph
                .search_native(
                    &NativeSpace {
                        vectors: &self.vectors,
                        boundary_factors: &self.boundary_factors,
                        dimensions: self.dimensions,
                        curvature,
                    },
                    query,
                    factor,
                    config,
                    navigation == PoincareNavigation::GeometryBq2,
                )
                .into_iter()
                .map(|(id, distance)| score_distance(id, distance))
                .collect(),
        }
    }

    /// 全量精确搜索，用于同一数据、几何和删除状态下的近邻真值。
    /// 返回分数与 `search` 相同：余弦相似度或负双曲测地距离，按降序排列。
    pub fn search_exact(&self, query: &[f32], top_k: usize) -> Result<Vec<(u64, f32)>> {
        if top_k == 0 {
            return Err(Error::InvalidSearchConfig("top_k 必须大于零"));
        }
        validate_vector(query, self.dimensions)?;
        let factor = self.metric.boundary_factor(query, 0)?;
        let mut candidates = Vec::with_capacity(self.len());
        for (slot, &id) in self.ids.iter().enumerate() {
            if !self.active[slot] {
                continue;
            }
            let vector = &self.vectors[slot * self.dimensions..(slot + 1) * self.dimensions];
            let score = match self.metric {
                Metric::Cosine => f64::from(cosine_similarity_f32(query, vector)),
                Metric::Poincare { curvature } => -poincare_distance(
                    poincare_proxy(query, vector, factor * self.boundary_factors[slot]),
                    curvature,
                ),
            };
            candidates.push((id, score));
        }
        // 仅对选出的 top_k 排序，避免精确真值排序整个百万文档库。
        if candidates.len() > top_k {
            candidates
                .select_nth_unstable_by(top_k, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            candidates.truncate(top_k);
        }
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        candidates
            .into_iter()
            .map(|(id, score)| {
                if score.is_finite() && (score as f32).is_finite() {
                    Ok((id, score as f32))
                } else {
                    Err(Error::DistanceOverflow)
                }
            })
            .collect()
    }
}
