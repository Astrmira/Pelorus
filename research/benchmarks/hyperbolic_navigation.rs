//! 原生双曲 ANN 基准：默认百万节点，独立查询，按精确双曲近邻统计召回。

use quiver_ann::{BuildConfig, Index, Metric, SearchConfig};
use rayon::prelude::*;
use std::collections::HashSet;
use std::error::Error;
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "research")]
fn observe_graph(
    index: &Index,
    strata: &[usize],
    output: Option<&String>,
) -> Result<Option<Vec<bool>>, Box<dyn Error>> {
    let graph = index.graph_diagnostics();
    let mut reachable = vec![false; index.len()];
    let mut out_degrees = Vec::with_capacity(index.len());
    let mut in_degrees = Vec::with_capacity(index.len());
    let mut radial = [[0usize; 5]; 3];
    let mut nodes_file = output
        .map(|p| std::fs::File::create(format!("{p}.nodes.csv")))
        .transpose()?
        .map(std::io::BufWriter::new);
    if let Some(file) = &mut nodes_file {
        writeln!(file, "节点标识,半径分组,出度,入度,入口可达")?;
    }
    for node in &graph.nodes {
        let i = usize::try_from(node.id)?;
        assert!(i < strata.len(), "基准节点标识与输入行不一致");
        reachable[i] = node.reachable;
        out_degrees.push(node.out_degree);
        in_degrees.push(node.in_degree);
        let r = &mut radial[strata[i]];
        r[0] += 1;
        r[1] += node.out_degree;
        r[2] += node.in_degree;
        r[3] += usize::from(node.reachable);
        r[4] += usize::from(node.in_degree == 0);
        if let Some(file) = &mut nodes_file {
            writeln!(
                file,
                "{},{},{},{},{}",
                node.id, strata[i], node.out_degree, node.in_degree, node.reachable
            )?;
        }
    }
    if let Some(file) = &mut nodes_file {
        file.flush()?;
    }
    out_degrees.sort_unstable();
    in_degrees.sort_unstable();
    let percentiles = |values: &[usize]| {
        [
            values[values.len() / 2],
            values[((values.len() as f64 * 0.95).ceil() as usize).saturating_sub(1)],
            values[((values.len() as f64 * 0.99).ceil() as usize).saturating_sub(1)],
            *values.last().unwrap(),
        ]
    };
    let out = percentiles(&out_degrees);
    let incoming = percentiles(&in_degrees);
    let entry = graph.entry_id.unwrap();
    let vector = index.vector(entry).unwrap();
    let entry_norm = vector
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    let entry_depth = 2.0 * entry_norm.atanh();
    let line = format!(
        "{{\"节点数\":{},\"边数\":{},\"入口标识\":{entry},\"入口测地半径\":{entry_depth},\"最大层级\":{},\"入口可达节点数\":{},\"弱连通分量数\":{},\"最大弱连通分量\":{},\"零出度节点\":{},\"零入度节点\":{},\"自环\":{},\"重复边\":{},\"越界边\":{},\"底层图CRC32\":{},\"出度分位数50_95_99_最大\":{out:?},\"入度分位数50_95_99_最大\":{incoming:?},\"径向分组_节点数_出度和_入度和_可达数_零入度数\":{radial:?}}}",
        index.len(),
        graph.edges,
        graph.max_level,
        graph.reachable_nodes,
        graph.weak_components,
        graph.largest_weak_component,
        graph.zero_out_degree,
        graph.zero_in_degree,
        graph.self_loops,
        graph.duplicate_edges,
        graph.invalid_edges,
        graph.topology_crc32
    );
    assert_eq!(
        graph.invalid_edges + graph.self_loops + graph.duplicate_edges,
        0,
        "图中存在非法边"
    );
    println!("图结构诊断：{line}");
    if let Some(output) = output {
        std::fs::write(format!("{output}.graph.json"), line)?;
    }
    // 只有单层图才能把该入口的底层可达性直接解释为搜索的结构上界。
    Ok((graph.max_level == 0).then_some(reachable))
}

fn linux_cpu_stat() -> (u64, u64, u64) {
    for path in ["/sys/fs/cgroup/cpu/cpu.stat", "/sys/fs/cgroup/cpu.stat"] {
        if let Ok(text) = std::fs::read_to_string(path) {
            let value = |key: &str| {
                text.lines()
                    .find_map(|l| {
                        l.strip_prefix(key)
                            .and_then(|s| s.trim().parse::<u64>().ok())
                    })
                    .unwrap_or(0)
            };
            let time = value("throttled_time");
            return (
                value("nr_periods"),
                value("nr_throttled"),
                if time > 0 {
                    time
                } else {
                    value("throttled_usec").saturating_mul(1000)
                },
            );
        }
    }
    (0, 0, 0)
}

struct Random(u64);
impl Random {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn points(&mut self, count: usize, dimensions: usize) -> (Vec<f32>, Vec<usize>) {
        let mut vectors = Vec::with_capacity(count * dimensions);
        let mut strata = Vec::with_capacity(count);
        for _ in 0..count {
            let start = vectors.len();
            while vectors.len() < start + dimensions {
                let magnitude = (-2.0 * self.uniform().ln()).sqrt();
                let angle = std::f64::consts::TAU * self.uniform();
                let (sin, cos) = angle.sin_cos();
                vectors.push((magnitude * cos) as f32);
                if vectors.len() < start + dimensions {
                    vectors.push((magnitude * sin) as f32);
                }
            }
            // 按测地半径均匀采样，不是等半径壳层，也不是双曲体积均匀分布。
            let depth = 0.05 + 5.95 * self.uniform();
            let radius = (depth / 2.0).tanh();
            let norm = vectors[start..]
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            for value in &mut vectors[start..] {
                *value = (f64::from(*value) * radius / norm) as f32;
            }
            strata.push((depth / 2.0).floor().min(2.0) as usize);
        }
        (vectors, strata)
    }
}

fn argument(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    match args.iter().position(|value| value == name) {
        Some(index) => Ok(args.get(index + 1).ok_or("参数缺少数值")?.parse()?),
        None => Ok(default),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let count = argument("--count", 1_000_000)?;
    let dimensions = argument("--dim", 64)?;
    let query_count = argument("--queries", 1000)?;
    let m = argument("--m", 16)?;
    let ef_construction = argument("--ef-construction", 128)?;
    let threads = argument("--threads", std::thread::available_parallelism()?.get())?;
    let truth_threads = argument("--truth-threads", threads)?;
    let ef_max = argument("--ef-max", 256)?;
    if threads == 0 || truth_threads == 0 || ef_max < 32 {
        return Err("构图与真值线程数必须大于零，最大搜索宽度至少为32".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    if count < 10 || dimensions == 0 || query_count == 0 || count.checked_mul(dimensions).is_none()
    {
        return Err("节点至少为10，维度及查询数须大于零，数据容量不能溢出".into());
    }
    let args = std::env::args().collect::<Vec<_>>();
    let output = args
        .iter()
        .position(|value| value == "--output")
        .map(|i| args.get(i + 1).ok_or("输出参数缺少路径"))
        .transpose()?;
    let mut file = output.map(std::fs::File::create).transpose()?;
    let started = Instant::now();
    let (vectors, _document_strata) = Random(20261112).points(count, dimensions);
    let (queries, strata) = Random(20261113).points(query_count, dimensions);
    let document_crc = crc32fast::hash(bytemuck::cast_slice(&vectors));
    let query_crc = crc32fast::hash(bytemuck::cast_slice(&queries));
    let ids = (0..count as u64).collect::<Vec<_>>();
    println!(
        "已生成{count}条文档及{query_count}条独立查询，维度{dimensions}，曲率-1；并行构图线程数{threads}。"
    );
    println!(
        "数据生成耗时：{:.3}秒；开始原生双曲构图。",
        started.elapsed().as_secs_f64()
    );
    let cpu_before = linux_cpu_stat();
    let build_started = Instant::now();
    let index = Index::build_with_metric(
        dimensions,
        &ids,
        &vectors,
        &BuildConfig {
            m,
            ef_construction,
            alpha: 1.2,
        },
        Metric::Poincare { curvature: 1.0 },
    )?;
    let build_seconds = build_started.elapsed().as_secs_f64();
    let cpu_after = linux_cpu_stat();
    let throttling = [
        cpu_after.0.saturating_sub(cpu_before.0),
        cpu_after.1.saturating_sub(cpu_before.1),
        cpu_after.2.saturating_sub(cpu_before.2),
    ];
    assert_eq!(index.len(), count, "实际索引节点数与请求不一致");
    drop(vectors);
    #[cfg(feature = "research")]
    let reachable = observe_graph(&index, &_document_strata, output)?;
    #[cfg(not(feature = "research"))]
    let reachable: Option<Vec<bool>> = None;
    println!(
        "构图完成：实际节点{}，耗时{build_seconds:.3}秒；计算精确近邻真值。",
        index.len()
    );
    let exact_started = Instant::now();
    let truth_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(truth_threads)
        .build()?;
    let truth = truth_pool.install(|| {
        queries
            .par_chunks_exact(dimensions)
            .map(|q| {
                index
                    .search_exact(q, 10)
                    .map(|rows| rows.into_iter().map(|row| row.0).collect::<HashSet<_>>())
            })
            .collect::<quiver_ann::Result<Vec<_>>>()
    })?;
    let exact_seconds = exact_started.elapsed().as_secs_f64();
    println!("精确近邻真值完成：耗时{exact_seconds:.3}秒；开始单线程多查询评测。");
    let truth_reachable = truth
        .iter()
        .map(|set| {
            reachable.as_ref().map_or(1.0, |mask| {
                set.iter().filter(|&&id| mask[id as usize]).count() as f64 / 10.0
            })
        })
        .collect::<Vec<_>>();
    let reachable_ceiling = if reachable.is_some() {
        (truth_reachable.iter().sum::<f64>() / query_count as f64).to_string()
    } else {
        "null".to_string()
    };
    let mut truth_hash = crc32fast::Hasher::new();
    for set in &truth {
        let mut ids = set.iter().copied().collect::<Vec<_>>();
        ids.sort_unstable();
        for id in ids {
            truth_hash.update(&id.to_le_bytes());
        }
    }
    let truth_crc = truth_hash.finalize();
    let mut cases = output
        .map(|p| std::fs::File::create(format!("{p}.queries.csv")))
        .transpose()?
        .map(std::io::BufWriter::new);
    if let Some(file) = &mut cases {
        writeln!(
            file,
            "查询序号,半径分组,搜索宽度,召回率10,返回数,真值可达比例"
        )?;
    }
    for ef_search in [32, 64, 128, 256, 512, 1024]
        .into_iter()
        .filter(|&ef| ef <= ef_max)
    {
        let config = SearchConfig {
            top_k: 10,
            ef_search,
            rerank_limit: None,
        };
        for q in queries.chunks_exact(dimensions).take(32) {
            index.search(q, &config)?;
        }
        let mut latencies = Vec::with_capacity(query_count);
        let mut recalls = [0.0; 3];
        let mut counts = [0usize; 3];
        let search_started = Instant::now();
        for (i, q) in queries.chunks_exact(dimensions).enumerate() {
            let query_started = Instant::now();
            let result = index.search(q, &config)?;
            latencies.push(query_started.elapsed().as_secs_f64() * 1000.0);
            let recall = result
                .iter()
                .filter(|row| truth[i].contains(&row.0))
                .count() as f64
                / 10.0;
            recalls[strata[i]] += recall;
            counts[strata[i]] += 1;
            if let Some(file) = &mut cases {
                writeln!(
                    file,
                    "{i},{},{ef_search},{recall},{},{}",
                    strata[i],
                    result.len(),
                    truth_reachable[i]
                )?;
            }
        }
        let seconds = search_started.elapsed().as_secs_f64();
        let recall = recalls.iter().sum::<f64>() / query_count as f64;
        let radial = std::array::from_fn::<_, 3, _>(|i| {
            if counts[i] > 0 {
                recalls[i] / counts[i] as f64
            } else {
                0.0
            }
        });
        latencies.sort_by(f64::total_cmp);
        let percentile =
            |p: f64| latencies[((query_count as f64 * p).ceil() as usize).saturating_sub(1)];
        let base_line = format!(
            "{{\"节点数\":{count},\"实际节点数\":{},\"维度\":{dimensions},\"查询数\":{query_count},\"文档种子\":20261112,\"查询种子\":20261113,\"曲率\":-1,\"导航\":\"原始坐标双曲距离\",\"径向分布\":\"测地半径0.05至6均匀采样\",\"构图线程数\":{threads},\"构图邻居数\":{m},\"构图搜索宽度\":{ef_construction},\"构图秒\":{build_seconds},\"真值秒\":{exact_seconds},\"搜索宽度\":{ef_search},\"召回率10\":{recall},\"每秒查询数\":{},\"延迟中位毫秒\":{},\"延迟99分位毫秒\":{},\"半径分组召回率\":{radial:?},\"半径分组查询数\":{counts:?}}}",
            index.len(),
            query_count as f64 / seconds,
            percentile(0.5),
            percentile(0.99),
        );
        let line = format!(
            "{},\"文档CRC32\":{document_crc},\"查询CRC32\":{query_crc},\"精确真值CRC32\":{truth_crc},\"真值线程数\":{truth_threads},\"入口可达召回上界\":{reachable_ceiling},\"构图期限数_限流期限数_累计限流纳秒\":{throttling:?}}}",
            &base_line[..base_line.len() - 1]
        );
        println!("{line}");
        if let Some(file) = &mut file {
            writeln!(file, "{line}")?;
            file.flush()?;
        }
    }
    if let Some(file) = &mut cases {
        file.flush()?;
    }
    Ok(())
}
