//! 真实 Cohere 百万数据上的几何感知构图、量化导航和同图精确导航对照。

#[cfg(feature = "research")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use quiver_ann::research::{HyperbolicStrategy, set_hyperbolic_strategy};
    use quiver_ann::{BuildConfig, Index, PoincareNavigation, SearchConfig};
    use rayon::prelude::*;
    use std::io::{Read, Write};
    use std::path::Path;
    use std::time::Instant;

    const DIMENSIONS: usize = 768;
    const COUNT: usize = 1_000_000;
    const QUERIES: usize = 1000;
    let arguments = std::env::args().collect::<Vec<_>>();
    let argument = |name: &str| -> Result<&str, Box<dyn std::error::Error>> {
        let i = arguments
            .iter()
            .position(|value| value == name)
            .ok_or("缺少命令行参数")?;
        Ok(arguments.get(i + 1).ok_or("参数缺少值")?)
    };
    let data = Path::new(argument("--data")?);
    let output = Path::new(argument("--output")?);
    let threads = argument("--threads")?.parse::<usize>()?;
    let ef_max = arguments
        .iter()
        .position(|value| value == "--ef-max")
        .map(|i| arguments.get(i + 1).ok_or("搜索预算参数缺少值"))
        .transpose()?
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1024);
    let ef_min = arguments
        .iter()
        .position(|value| value == "--ef-min")
        .map(|i| arguments.get(i + 1).ok_or("最小搜索预算参数缺少值"))
        .transpose()?
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(32);
    let mode_tag = arguments
        .iter()
        .position(|value| value == "--mode")
        .map(|i| arguments.get(i + 1).ok_or("消融模式参数缺少值"))
        .transpose()?
        .map(String::as_str)
        .unwrap_or("a8");
    let strategy = match mode_tag {
        "a8" | "full" => HyperbolicStrategy::Full,
        "a7" => HyperbolicStrategy::FixedShellCompass,
        "a6" => HyperbolicStrategy::CalibratedCompass,
        "a5" => HyperbolicStrategy::PolarRouting,
        "a4" => HyperbolicStrategy::DualFiberExpansion,
        _ => return Err("未知 --mode 参数，可选值: a8 | a7 | a6 | a5 | a4".into()),
    };
    set_hyperbolic_strategy(strategy);
    if !(32..=65536).contains(&ef_max) || ef_min > ef_max || ef_min < 32 {
        return Err("搜索宽度区间必须在32至65536之间且下限不大于上限".into());
    }
    if threads == 0 {
        return Err("线程数必须大于零".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let read_vectors = |name: &str, rows: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(data.join(name))?;
        if file.metadata()?.len() != (rows * DIMENSIONS * 4) as u64 {
            return Err("真实向量文件尺寸不符合约定".into());
        }
        let mut vectors = vec![0.0f32; rows * DIMENSIONS];
        file.read_exact(bytemuck::cast_slice_mut(&mut vectors))?;
        if !cfg!(target_endian = "little") {
            for value in &mut vectors {
                *value = f32::from_bits(value.to_bits().swap_bytes());
            }
        }
        Ok(vectors)
    };
    let vectors = read_vectors("documents.f32", COUNT)?;
    let queries = read_vectors("queries.f32", QUERIES)?;
    let truth_bytes = std::fs::read(data.join("truth.u64"))?;
    if truth_bytes.len() != QUERIES * 10 * 8 {
        return Err("精确真值文件尺寸不符合约定".into());
    }
    let truth = truth_bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| u64::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    if truth.iter().any(|&id| id >= COUNT as u64) {
        return Err("精确真值含越界标识".into());
    }
    let doc_crc = crc32fast::hash(bytemuck::cast_slice(&vectors));
    let query_crc = crc32fast::hash(bytemuck::cast_slice(&queries));
    let truth_crc = crc32fast::hash(&truth_bytes);
    println!(
        "已载入真实 Cohere 百万条、768维向量及1000条查询，模式={mode_tag}，使用{threads}线程几何感知构图"
    );
    let started = Instant::now();
    let index = Index::build_poincare(
        DIMENSIONS,
        &(0..COUNT as u64).collect::<Vec<_>>(),
        &vectors,
        &BuildConfig {
            m: 16,
            ef_construction: 128,
            alpha: 1.2,
        },
        1.0,
        PoincareNavigation::GeometryBq2,
    )?;
    let build_seconds = started.elapsed().as_secs_f64();
    drop(vectors);
    assert_eq!(index.len(), COUNT);
    println!("几何量化构图完成，耗时{build_seconds:.3}秒，正在交叉核验独立真值");
    (0..QUERIES)
        .step_by(100)
        .collect::<Vec<_>>()
        .par_iter()
        .for_each(|&i| {
            let exact = index
                .search_exact(&queries[i * DIMENSIONS..(i + 1) * DIMENSIONS], 10)
                .unwrap();
            let actual = exact.iter().map(|row| row.0).collect::<Vec<_>>();
            assert_eq!(
                actual,
                truth[i * 10..(i + 1) * 10],
                "独立FP64真值与坐标差内核不一致"
            );
        });
    let graph = index.graph_diagnostics();
    assert_eq!(
        graph.invalid_edges + graph.self_loops + graph.duplicate_edges,
        0
    );
    let reachable = graph
        .nodes
        .iter()
        .map(|node| node.reachable)
        .collect::<Vec<_>>();
    let ceiling =
        truth.iter().filter(|&&id| reachable[id as usize]).count() as f64 / truth.len() as f64;
    let graph_line = format!(
        "{{\"节点数\":{},\"边数\":{},\"最大层级\":{},\"入口可达节点数\":{},\"零入度节点\":{},\"弱连通分量数\":{},\"底层图CRC32\":{},\"真值可达召回上界\":{},\"导航有效载荷字节数\":{}}}",
        graph.nodes.len(),
        graph.edges,
        graph.max_level,
        graph.reachable_nodes,
        graph.zero_in_degree,
        graph.weak_components,
        graph.topology_crc32,
        ceiling,
        index.navigation_storage_bytes()
    );
    std::fs::write(output.with_extension("graph.json"), &graph_line)?;
    println!("图结构诊断 {graph_line}");
    drop(graph);
    let mut report = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
    );
    let mut cases = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.with_extension("queries.csv"))?,
    );
    writeln!(cases, "导航,搜索宽度,查询序号,召回率10,搜索毫秒")?;
    for (round, ef) in [
        32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    ]
    .into_iter()
    .filter(|&ef| ef >= ef_min && ef <= ef_max)
    .enumerate()
    {
        let methods = if round % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        };
        for quantized in methods {
            let method = if quantized {
                "几何BQ2导航"
            } else {
                "同图精确导航"
            };
            let config = SearchConfig {
                top_k: 10,
                ef_search: ef,
                rerank_limit: None,
            };
            let search = |query: &[f32]| {
                if quantized {
                    index.search(query, &config)
                } else {
                    index.search_poincare_exact_navigation(query, &config)
                }
            };
            for query in queries.as_chunks::<DIMENSIONS>().0.iter().take(32) {
                search(query)?;
            }
            let mut latencies = Vec::with_capacity(QUERIES);
            let mut recall_sum = 0.0;
            for (i, query) in queries.as_chunks::<DIMENSIONS>().0.iter().enumerate() {
                let started = Instant::now();
                let result = search(query)?;
                let seconds = started.elapsed().as_secs_f64();
                latencies.push(seconds);
                let expected = &truth[i * 10..(i + 1) * 10];
                let recall = result
                    .iter()
                    .filter(|row| expected.contains(&row.0))
                    .count() as f64
                    / 10.0;
                recall_sum += recall;
                assert_eq!(result.len(), 10);
                assert!(result.windows(2).all(|pair| pair[0].1 >= pair[1].1));
                writeln!(cases, "{method},{ef},{i},{recall},{}", seconds * 1000.0)?;
            }
            let qps = QUERIES as f64 / latencies.iter().sum::<f64>();
            latencies.sort_by(f64::total_cmp);
            let line = format!(
                "{{\"数据集\":\"Cohere-768D-1M\",\"导航\":\"{method}\",\"实际节点数\":{COUNT},\"维度\":{DIMENSIONS},\"查询数\":{QUERIES},\"构图线程数\":{threads},\"构图秒\":{build_seconds},\"搜索宽度\":{ef},\"召回率10\":{},\"每秒查询数\":{qps},\"延迟中位毫秒\":{},\"延迟99分位毫秒\":{},\"文档CRC32\":{doc_crc},\"查询CRC32\":{query_crc},\"精确真值CRC32\":{truth_crc},\"独立真值交叉核验查询数\":10}}",
                recall_sum / QUERIES as f64,
                latencies[499] * 1000.0,
                latencies[989] * 1000.0
            );
            println!("{line}");
            writeln!(report, "{line}")?;
            report.flush()?;
            cases.flush()?;
        }
    }
    Ok(())
}

#[cfg(not(feature = "research"))]
fn main() {
    eprintln!("该基准需要启用 research 特性");
    std::process::exit(1);
}
