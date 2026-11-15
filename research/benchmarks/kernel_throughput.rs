//! 测量明确工作集下的热缓存、顺序及随机访问位核性能。
#[cfg(feature = "research")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds = args
        .get(1)
        .map(|v| v.parse().expect("轮次必须为整数"))
        .unwrap_or(1_000_000);
    for nodes in [64usize, 4096, 131072, 1048576] {
        for random in [false, true] {
            for repeat in 0..3 {
                for (kernel, name, ns) in quiver_ann::research::benchmark_sign_magnitude_workload(
                    nodes,
                    rounds,
                    random,
                    12345 + repeat,
                ) {
                    println!(
                        "{}",
                        serde_json::json!({"节点数":nodes,"签名工作集字节":nodes*192,"随机访问":random,
                        "重复":repeat,"位核":kernel,"名称":name,"评估次数":rounds,"单次纳秒":ns})
                    );
                }
            }
        }
    }
}

#[cfg(not(feature = "research"))]
fn main() {
    eprintln!("位核复测需要启用 research 特性");
    std::process::exit(1);
}
