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

/// 变曲率流形实验原型入口（仅在启用 `research` feature 时编译，聚焦 Tier 2 乘积流形与 Tier 3 径向翘曲乘积流形探索，暂不转正）。
pub mod var_curvature {
    pub use crate::var_curvature::{
        GraphBuildConfig, ProductBaselineIndex, ProductBaselineMode, ProductBlockSmIndex,
        ProductFactor, ProductSingleBallIndex, SearchOutcome, Tier3Method, VarGraph,
        WarpedManifoldIndex, WarpedProfile, WarpedProxyTable, clairaut_distance,
        exact_product_distance_sq, generate_product_synthetic, generate_warped_synthetic,
        product_ground_truth, warped_ground_truth,
    };
}
