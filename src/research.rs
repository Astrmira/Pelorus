//! 用于性能研究、表示对照与计时外诊断的配置入口。
//!
//! 这些全局配置须在构图和并发查询启动前设置，使用期间保持固定。

pub use crate::graph::{
    ExecutionConfig, GeometryComponents, GraphDiagnostics, HyperbolicStrategy, NodeDiagnostics,
    RecallConfig, SearchLossStats, SearchStageProfile, SketchFilter, StorageConfig,
    TraversalProfile, benchmark_sign_magnitude_kernels, benchmark_sign_magnitude_workload,
    execution_config, hyperbolic_strategy, recall_config, set_execution_config,
    set_geometry_components, set_hyperbolic_strategy, set_recall_config, set_storage_config,
    storage_config,
};
pub use crate::quantization::set_asym_sm_lut_override;
