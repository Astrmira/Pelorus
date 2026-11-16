//! 插入、删除、压缩与重建。

use super::*;
impl Index {
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        validate_vector(vector, self.dimensions)?;
        let factor = self.metric.boundary_factor(vector, self.ids.len())?;
        if self.id_to_slot.contains_key(&id) {
            return Err(Error::DuplicateId(id));
        }
        if self.ids.len() >= u32::MAX as usize {
            return Err(Error::CapacityExceeded);
        }
        self.vectors
            .len()
            .checked_add(self.dimensions)
            .ok_or(Error::CapacityExceeded)?;

        let encoded = self
            .rerank
            .as_ref()
            .map(|store| store.encode(vector, self.metric, self.ids.len()))
            .transpose()?;
        self.vectors.reserve(self.dimensions)?;
        if let Some(store) = &mut self.rerank {
            store.bytes.reserve(store.stride)?;
            store
                .prepared
                .try_reserve(1)
                .map_err(|_| Error::CapacityExceeded)?;
        }
        self.ids
            .try_reserve(1)
            .map_err(|_| Error::CapacityExceeded)?;
        self.active
            .try_reserve(1)
            .map_err(|_| Error::CapacityExceeded)?;
        self.id_to_slot
            .try_reserve(1)
            .map_err(|_| Error::CapacityExceeded)?;
        self.boundary_factors
            .try_reserve(1)
            .map_err(|_| Error::CapacityExceeded)?;
        let slot = self.ids.len();
        self.vectors.append(vector);
        match self.metric {
            Metric::Cosine => self.graph.insert(vector, id, slot, &mut self.next_seed),
            Metric::Poincare { curvature } => {
                self.boundary_factors.push(factor);
                let (vectors, delta) = self.vectors.slices();
                self.graph.insert_native(
                    &NativeSpace {
                        vectors,
                        delta,
                        rerank: None,
                        boundary_factors: &self.boundary_factors,
                        dimensions: self.dimensions,
                        curvature,
                    },
                    id,
                    slot,
                    &mut self.next_seed,
                    self.navigation == PoincareNavigation::GeometryBq2,
                );
            }
        }
        self.graph.dirty_count_inc();
        if let Some(store) = &mut self.rerank {
            let (bytes, prepared) = encoded.as_ref().unwrap();
            store.bytes.append(bytes);
            store.prepared.push(*prepared);
        }
        self.ids.push(id);
        self.active.push(true);
        self.id_to_slot.insert(id, slot);
        Ok(())
    }

    pub fn remove(&mut self, id: u64) -> bool {
        let Some(slot) = self.id_to_slot.remove(&id) else {
            return false;
        };
        self.active[slot] = false;
        self.graph.soft_delete(id)
    }

    /// 丢弃软删除数据并重新构建图，返回实际清理的节点数。
    pub fn compact(&mut self) -> Result<usize> {
        let removed = self.ids.len() - self.len();
        self.rebuild()?;
        Ok(removed)
    }

    /// 使用当前构建配置重新构建所有活跃节点。
    pub fn rebuild(&mut self) -> Result<()> {
        let config = self.config;
        self.rebuild_with_config(&config)
    }

    /// 使用新配置重新构建所有活跃节点。
    pub fn rebuild_with_config(&mut self, config: &BuildConfig) -> Result<()> {
        validate_build_config(self.dimensions, config)?;
        let mut ids = Vec::with_capacity(self.len());
        let mut vectors = Vec::with_capacity(
            self.len()
                .checked_mul(self.dimensions)
                .ok_or(Error::CapacityExceeded)?,
        );
        for (slot, (&id, &active)) in self.ids.iter().zip(&self.active).enumerate() {
            if active {
                ids.push(id);
                vectors.extend_from_slice(self.vectors.row(slot, self.dimensions));
            }
        }
        let preserve_residual = self.graph.residual_cone_count() > 0;
        let rerank_config = self.rerank_config();
        let storage_mode = self.storage_mode;
        let ssd_config = self.ssd_cache_config();
        let mut rebuilt = Self::build_impl_with_options(
            self.dimensions,
            &ids,
            &vectors,
            config,
            self.metric,
            self.navigation,
            preserve_residual,
        )?;
        rebuilt.set_rerank_config(rerank_config)?;
        rebuilt.set_storage_mode(storage_mode)?;
        if let Some(config) = ssd_config {
            rebuilt.set_ssd_cache_config(config)?;
        }
        *self = rebuilt;
        Ok(())
    }
}
