//! 索引文件格式、校验与原子保存。

use super::*;
pub(crate) const FILE_MAGIC: &[u8; 8] = b"QUIVERIX";
pub(crate) const FILE_VERSION: u32 = 3;
pub(crate) const HEADER_LEN: usize = 8 + 4 + 8 + 4;
impl Index {
    /// 将完整 owned 索引保存为带校验和的单文件，并以临时文件原子替换目标。
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut payload = Vec::new();
        // v1 记录余弦，v2 记录精确双曲距离，v3 额外记录几何量化导航模式（1=标准 SM，2=切锥正交残差 SM）。
        let version = match self.metric {
            Metric::Cosine => 1,
            Metric::Poincare { curvature } => {
                payload.extend_from_slice(&1u32.to_le_bytes());
                payload.extend_from_slice(&curvature.to_bits().to_le_bytes());
                if self.navigation == PoincareNavigation::GeometryBq2 {
                    let nav_tag = if self.graph.residual_cone_count() > 0 {
                        2u32
                    } else {
                        1u32
                    };
                    payload.extend_from_slice(&nav_tag.to_le_bytes());
                    FILE_VERSION
                } else {
                    2
                }
            }
        };
        payload.extend_from_slice(&(self.dimensions as u64).to_le_bytes());
        payload.extend_from_slice(&(self.len() as u64).to_le_bytes());
        payload.extend_from_slice(&(self.config.m as u64).to_le_bytes());
        payload.extend_from_slice(&(self.config.ef_construction as u64).to_le_bytes());
        payload.extend_from_slice(&self.config.alpha.to_bits().to_le_bytes());
        for (&id, &active) in self.ids.iter().zip(&self.active) {
            if active {
                payload.extend_from_slice(&id.to_le_bytes());
            }
        }
        for (slot, &active) in self.active.iter().enumerate() {
            if active {
                let start = slot * self.dimensions;
                for &value in &self.vectors[start..start + self.dimensions] {
                    payload.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }

        let mut bytes = Vec::with_capacity(
            HEADER_LEN
                .checked_add(payload.len())
                .ok_or(Error::CapacityExceeded)?,
        );
        bytes.extend_from_slice(FILE_MAGIC);
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let path = path.as_ref();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
        temporary.write_all(&bytes)?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(path)
            .map_err(|error| Error::from(error.error))?;
        Ok(())
    }

    /// 从 owned 单文件加载并校验数据，然后重建内存图。
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
        if bytes.len() < HEADER_LEN || &bytes[..8] != FILE_MAGIC {
            return Err(Error::CorruptIndex("文件头无效"));
        }
        let version = read_u32(&bytes, 8)?;
        if !(1..=FILE_VERSION).contains(&version) {
            return Err(Error::UnsupportedVersion(version));
        }
        let payload_len =
            usize::try_from(read_u64(&bytes, 12)?).map_err(|_| Error::CapacityExceeded)?;
        let expected_len = HEADER_LEN
            .checked_add(payload_len)
            .ok_or(Error::CapacityExceeded)?;
        if bytes.len() != expected_len {
            return Err(Error::CorruptIndex("文件长度不匹配"));
        }
        let checksum = read_u32(&bytes, 20)?;
        let payload = &bytes[HEADER_LEN..];
        if crc32fast::hash(payload) != checksum {
            return Err(Error::CorruptIndex("校验和不匹配"));
        }

        let mut cursor = 0usize;
        let metric = if version == 1 {
            Metric::Cosine
        } else {
            let tag = read_payload_u32(payload, &mut cursor)?;
            let curvature = f64::from_bits(read_payload_u64(payload, &mut cursor)?);
            if tag != 1 {
                return Err(Error::CorruptIndex("未知几何类型"));
            }
            let metric = Metric::Poincare { curvature };
            metric.validate()?;
            metric
        };
        let (navigation, force_residual) = if version == 3 {
            match read_payload_u32(payload, &mut cursor)? {
                1 => (PoincareNavigation::GeometryBq2, false),
                2 => (PoincareNavigation::GeometryBq2, true),
                _ => return Err(Error::CorruptIndex("未知双曲导航模式")),
            }
        } else {
            (PoincareNavigation::Exact, false)
        };
        let dimensions = read_payload_usize(payload, &mut cursor)?;
        let count = read_payload_usize(payload, &mut cursor)?;
        let m = read_payload_usize(payload, &mut cursor)?;
        let ef_construction = read_payload_usize(payload, &mut cursor)?;
        let alpha = f32::from_bits(read_payload_u32(payload, &mut cursor)?);
        let config = BuildConfig {
            m,
            ef_construction,
            alpha,
        };
        validate_build_config(dimensions, &config)?;
        if count > u32::MAX as usize {
            return Err(Error::CapacityExceeded);
        }
        let vector_values = count
            .checked_mul(dimensions)
            .ok_or(Error::CapacityExceeded)?;
        let body_len = count
            .checked_mul(8)
            .and_then(|ids_len| {
                vector_values
                    .checked_mul(4)
                    .and_then(|v| ids_len.checked_add(v))
            })
            .ok_or(Error::CapacityExceeded)?;
        if payload.len().saturating_sub(cursor) != body_len {
            return Err(Error::CorruptIndex("数据区长度不匹配"));
        }

        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(read_payload_u64(payload, &mut cursor)?);
        }
        let mut vectors = Vec::with_capacity(vector_values);
        for _ in 0..vector_values {
            vectors.push(f32::from_bits(read_payload_u32(payload, &mut cursor)?));
        }
        Self::build_impl_with_options(
            dimensions,
            &ids,
            &vectors,
            &config,
            metric,
            navigation,
            force_residual,
        )
    }
}
pub(crate) fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let raw: [u8; 4] = bytes
        .get(offset..offset + 4)
        .ok_or(Error::CorruptIndex("字段被截断"))?
        .try_into()
        .map_err(|_| Error::CorruptIndex("字段无效"))?;
    Ok(u32::from_le_bytes(raw))
}
pub(crate) fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let raw: [u8; 8] = bytes
        .get(offset..offset + 8)
        .ok_or(Error::CorruptIndex("字段被截断"))?
        .try_into()
        .map_err(|_| Error::CorruptIndex("字段无效"))?;
    Ok(u64::from_le_bytes(raw))
}
pub(crate) fn read_payload_u32(payload: &[u8], cursor: &mut usize) -> Result<u32> {
    let value = read_u32(payload, *cursor)?;
    *cursor = cursor.checked_add(4).ok_or(Error::CapacityExceeded)?;
    Ok(value)
}
pub(crate) fn read_payload_u64(payload: &[u8], cursor: &mut usize) -> Result<u64> {
    let value = read_u64(payload, *cursor)?;
    *cursor = cursor.checked_add(8).ok_or(Error::CapacityExceeded)?;
    Ok(value)
}
pub(crate) fn read_payload_usize(payload: &[u8], cursor: &mut usize) -> Result<usize> {
    usize::try_from(read_payload_u64(payload, cursor)?).map_err(|_| Error::CapacityExceeded)
}
