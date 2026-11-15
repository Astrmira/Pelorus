//! QuIVer 独立 ANN 引擎。
//!
//! 支持 BQ2 余弦检索，以及精确距离和几何感知 BQ2 两种原生双曲路径。

#![allow(clippy::needless_range_loop)]
#![allow(clippy::unnecessary_sort_by)]
#![allow(dead_code)]

mod config;
#[cfg(feature = "research")]
mod direction;
mod geometry;
mod graph;
mod index;
mod metric;
mod quantization;
mod vector;

pub use config::{BuildConfig, SearchConfig};
pub use geometry::PoincareNavigation;
pub use index::{Error, Index, Result};
pub use metric::Metric;
#[cfg(feature = "research")]
pub mod research;
pub(crate) use vector::VectorType;
