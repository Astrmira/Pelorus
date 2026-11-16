//! 冷向量存储与重排表示的公开配置。

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StorageMode {
    #[default]
    /// 重排记录常驻堆；启用编码时，原始向量使用文件后备供维护按需访问。
    Memory,
    /// 图与导航状态常驻内存，原始及重排向量通过只读映射按需分页。
    Mmap,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RerankPrecision {
    #[default]
    Fp32,
    Fp16,
    Bf16,
    Sq8,
    Sq4,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RerankRepresentation {
    #[default]
    Cartesian,
    /// 单独保存原始 FP64 半径平方，方向在解码后以 FP64 重新归一化。
    Polar,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScalarQuantization {
    #[default]
    Symmetric,
    /// 从活跃库向量训练共享逐维均值，插入时沿用已冻结的均值。
    Centered,
    /// 每条向量的每组独立保存 FP32 中点偏置与尺度，支持非整组尾部。
    Affine { group_size: usize },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RerankConfig {
    pub precision: RerankPrecision,
    pub representation: RerankRepresentation,
    pub scalar_quantization: ScalarQuantization,
    /// 为笛卡尔编码增加 FP32 范数恢复系数。
    pub norm_correction: bool,
}

impl RerankConfig {
    pub(crate) fn validate(self) -> Result<()> {
        let scalar = matches!(self.precision, RerankPrecision::Sq8 | RerankPrecision::Sq4);
        if !scalar && self.scalar_quantization != ScalarQuantization::Symmetric {
            return Err(Error::InvalidRerankConfig(
                "中心化与分组配置只适用于 SQ 编码",
            ));
        }
        if matches!(
            self.scalar_quantization,
            ScalarQuantization::Affine { group_size: 0 }
        ) {
            return Err(Error::InvalidRerankConfig("量化组大小必须大于零"));
        }
        if self.norm_correction && self.representation == RerankRepresentation::Polar {
            return Err(Error::InvalidRerankConfig("极坐标表示已独立保存原始半径"));
        }
        Ok(())
    }
}

/// 堆载荷与逻辑映射长度分开统计；映射驻留页由操作系统管理。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VectorStorageStats {
    pub canonical_heap_bytes: usize,
    pub canonical_mapped_bytes: usize,
    pub rerank_heap_bytes: usize,
    pub rerank_mapped_bytes: usize,
    pub shared_metadata_bytes: usize,
    pub rerank_metadata_bytes: usize,
    pub rerank_record_bytes: usize,
}

/// Linux 直接读取后端的共享页缓存与在途缓冲预算。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SsdIoDriver {
    #[default]
    IoUring,
    Pread,
}

/// Linux直接读取的缓存、缓冲预算和驱动配置。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsdCacheConfig {
    pub cache_bytes: usize,
    pub inflight_bytes: usize,
    /// 阻塞Pread驱动的工作线程数；IoUring使用一个独占ring的调度线程。
    pub io_workers: usize,
    pub prefetch_window: usize,
    pub driver: SsdIoDriver,
}

impl Default for SsdCacheConfig {
    fn default() -> Self {
        Self {
            cache_bytes: 128 * 1024 * 1024,
            inflight_bytes: 32 * 1024 * 1024,
            io_workers: 1,
            prefetch_window: 32,
            driver: SsdIoDriver::IoUring,
        }
    }
}

impl SsdCacheConfig {
    pub(crate) fn validate(self) -> Result<()> {
        if !self.cache_bytes.is_multiple_of(4096)
            || !self.inflight_bytes.is_multiple_of(4096)
            || self.inflight_bytes < 4096
            || self.io_workers == 0
            || self.prefetch_window > 128
            || self.cache_bytes.checked_add(self.inflight_bytes).is_none()
        {
            return Err(Error::InvalidRerankConfig(
                "SSD 缓存及在途预算须为4096字节整数倍，在途预算至少一页，工作线程须为正，预取窗口至多128",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SsdIoStats {
    pub driver: SsdIoDriver,
    pub io_threads: usize,
    pub queue_depth: usize,
    pub pending_peak_bytes: usize,
    pub prefetch_skipped: u64,
    pub uring_submitted: u64,
    pub uring_completed: u64,
    pub uring_max_inflight: u64,
    pub uring_submit_batches: u64,
    pub uring_completion_batches: u64,
    pub uring_completion_batch_max: u64,
    pub uring_enter_calls: u64,
    pub uring_enter_eintr: u64,
    pub ring_errors: u64,
    pub registered_buffer_bytes: u64,
    pub ring_mapping_bytes: u64,
    pub io_wq_affinity_registered: bool,
    pub io_wq_workers_registered: bool,
    pub read_calls: u64,
    pub read_bytes: u64,
    pub read_errors: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub pending_hits: u64,
    pub query_page_hits: u64,
    pub prefetch_requests: u64,
    pub prefetch_used: u64,
    pub prefetch_wasted: u64,
    pub evicted_pages: u64,
    pub cache_current_bytes: usize,
    pub cache_peak_bytes: usize,
    pub cache_capacity_bytes: usize,
    pub pool_allocated_bytes: usize,
    pub pool_live_bytes: usize,
    pub pool_peak_live_bytes: usize,
    pub pool_limit_bytes: usize,
}
