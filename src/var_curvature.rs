//! 变曲率几何原型模块（Tier 2 混合/乘积常曲率流形 + Tier 3 径向变曲率翘曲乘积流形）。
//!
//! 仅在 `research` 特性下编译，用于 50k 合成流形上的纯工程探索与对照验证，不影响主路径生产契约。

use crate::geometry::sm_calibrated_cosine_fast;
use crate::metric::{norm_squared_f64, poincare_distance, poincare_proxy};
use crate::quantization::{AsymQuerySm, Bq2Signature, Bq2Store, SmCodeParams, sm_dot_counts_raw};
use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq)]
struct OrdDist(f64);

impl Eq for OrdDist {}

impl PartialOrd for OrdDist {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdDist {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// 搜索统计与结果封装（用于对照阶段开销、候选裁剪与精确重排计数）。
#[derive(Clone, Debug)]
pub struct SearchOutcome {
    pub results: Vec<(usize, f64)>,
    pub exact_reranks: usize,
    pub radial_rejections: usize,
    pub expansions: usize,
}

/// 确定性 PCG/LCG 伪随机数生成器，用于独立可复现的合成数据集生成。
struct DetRng(u64);

impl DetRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    #[inline]
    fn uniform01(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[inline]
    fn gauss(&mut self) -> f64 {
        let u1 = self.uniform01().clamp(1e-12, 1.0 - 1e-12);
        let u2 = self.uniform01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[inline]
fn dot_f32_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;
    let n = a.len().min(b.len());
    let chunks = n / 4 * 4;
    let mut i = 0;
    while i < chunks {
        s0 += f64::from(a[i]) * f64::from(b[i]);
        s1 += f64::from(a[i + 1]) * f64::from(b[i + 1]);
        s2 += f64::from(a[i + 2]) * f64::from(b[i + 2]);
        s3 += f64::from(a[i + 3]) * f64::from(b[i + 3]);
        i += 4;
    }
    while i < n {
        s0 += f64::from(a[i]) * f64::from(b[i]);
        i += 1;
    }
    (s0 + s1) + (s2 + s3)
}

#[inline]
fn normalize_dir(v: &[f64]) -> Vec<f64> {
    let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-12);
    v.iter().map(|&x| x / norm).collect()
}

/// 从浅中内层候选池（`scores` 最小的 30% 区间内均匀抽取最多 256 个节点）中贪心选取 `k` 个彼此方向最分散的枢纽入口。
fn pick_diverse_hubs(vectors: &[f32], total_dim: usize, scores: &[f64], k: usize) -> Vec<u32> {
    let n = scores.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let mut rank: Vec<(f64, usize)> = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect();
    rank.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let inner_cap = (n * 3 / 10).max(k).min(n);
    let sample_count = inner_cap.min(256);
    let mut sample_ids = Vec::with_capacity(sample_count);
    for s in 0..sample_count {
        sample_ids.push(rank[(s * inner_cap) / sample_count].1);
    }
    let unit_vecs: Vec<Vec<f64>> = sample_ids
        .iter()
        .map(|&id| {
            let v: Vec<f64> = vectors[id * total_dim..(id + 1) * total_dim]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            normalize_dir(&v)
        })
        .collect();

    let mut chosen = Vec::with_capacity(k.min(sample_count));
    chosen.push(0usize);
    while chosen.len() < k.min(sample_count) {
        let mut best_idx = 0usize;
        let mut best_max_cos = f64::INFINITY;
        for (i, ui) in unit_vecs.iter().enumerate() {
            if chosen.contains(&i) {
                continue;
            }
            let max_c = chosen
                .iter()
                .map(|&c| {
                    ui.iter()
                        .zip(&unit_vecs[c])
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
                        .abs()
                })
                .fold(f64::NEG_INFINITY, f64::max);
            if max_c < best_max_cos {
                best_max_cos = max_c;
                best_idx = i;
            }
        }
        chosen.push(best_idx);
    }
    chosen.into_iter().map(|p| sample_ids[p] as u32).collect()
}

// ═════════════════════════════════════════════════════════════════════════════
//  通用单层 Vamana 图容器（增量小批量插入 + 全局 α-RNG 细化，确保公平对比）
// ═════════════════════════════════════════════════════════════════════════════

#[inline(always)]
fn visit_check_set(words: &mut [u64], idx: usize) -> bool {
    let word = &mut words[idx >> 6];
    let mask = 1u64 << (idx & 63);
    let was = (*word & mask) != 0;
    *word |= mask;
    was
}

/// 黎曼四向罗盘剪枝几何上下文（提供多径向 Stem-Excess 下界与纯球面角度损失）。
struct QuadriPruneSpec<'a> {
    depths: &'a [f64],
    radial_dist_fn: &'a (dyn Fn(usize, usize) -> f64 + Sync),
    angular_loss_fn: &'a (dyn Fn(usize, usize) -> f64 + Sync),
    iso_tol: f64,
    elev_min: f64,
    elev_step: f64,
    geo_fraction: f64,
    ang_guard_delta: f64,
    elev_ang_delta: f64,
    iso_score_uses_geodesic: bool,
    bridge_cap: usize,
}

/// 通用 L0 单层导航图。
pub struct VarGraph {
    pub neighbors: Vec<Vec<u32>>,
    pub entry_points: Vec<u32>,
    pub m_max: usize,
}

impl VarGraph {
    /// 增量波次 Vamana 构图 + Pass-2 全局 α-RNG 细化。
    ///
    /// 注：所有 `dist_fn` 均为平方距离/距离代理，因此剪枝判定采用 `alpha * alpha` 与距离空间 `alpha` 对齐。
    pub fn build<F>(
        n: usize,
        m_max: usize,
        ef_construction: usize,
        alpha: f64,
        entry_points: Vec<u32>,
        dist_fn: F,
    ) -> Self
    where
        F: Fn(usize, usize) -> f64 + Sync,
    {
        Self::build_with_quadri(
            n,
            m_max,
            ef_construction,
            alpha,
            entry_points,
            dist_fn,
            None,
        )
    }

    fn build_with_quadri<F>(
        n: usize,
        m_max: usize,
        ef_construction: usize,
        alpha: f64,
        entry_points: Vec<u32>,
        dist_fn: F,
        quadri: Option<&QuadriPruneSpec<'_>>,
    ) -> Self
    where
        F: Fn(usize, usize) -> f64 + Sync,
    {
        if n == 0 {
            return Self {
                neighbors: Vec::new(),
                entry_points: Vec::new(),
                m_max,
            };
        }

        let entries: Vec<u32> = if entry_points.is_empty() {
            vec![0]
        } else {
            entry_points
        };

        // DiskANN 风格：先以随机 M-正则有向拓展图初始化全部 N 个节点，保证全图（含多聚簇外层）初始全连通
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = DetRng::new(0x5EED_CAFE);
        for i in (1..n).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            order.swap(i, j);
        }
        let deg_init = m_max.min(n - 1);
        let mut adj: Vec<Vec<u32>> = (0..n)
            .map(|u| {
                let mut edges = Vec::with_capacity(deg_init);
                let mut attempts = 0usize;
                while edges.len() < deg_init && attempts < deg_init * 4 {
                    let cand = (rng
                        .next_u64()
                        .wrapping_add((u as u64).wrapping_mul(2654435761))
                        % (n as u64)) as usize;
                    attempts += 1;
                    if cand != u && !edges.contains(&(cand as u32)) {
                        edges.push(cand as u32);
                    }
                }
                edges
            })
            .collect();

        // 增量波次在随机骨架上做局部到全局平滑细化（α = 1.0 + 双向回插）
        let mut cursor = 0usize;
        let mut batch_cap = 1024usize;
        while cursor < n {
            let end = (cursor + batch_cap).min(n);
            let batch = &order[cursor..end];
            let updates: Vec<(usize, Vec<u32>)> = batch
                .par_iter()
                .map(|&node| {
                    let local_seeds = &adj[node][..adj[node].len().min(4)];
                    let mut cands = beam_collect_internal(
                        &adj,
                        &entries,
                        local_seeds,
                        ef_construction,
                        n,
                        Some(node),
                        |other| dist_fn(node, other),
                    );
                    for &nb in &adj[node] {
                        if nb as usize != node {
                            cands.push((OrdDist(dist_fn(node, nb as usize)), nb));
                        }
                    }
                    cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                    cands.dedup_by_key(|x| x.1);
                    let pruned = prune_dispatch(node, &cands, m_max, 1.0, &dist_fn, quadri);
                    (node, pruned)
                })
                .collect();

            let mut touched: Vec<(usize, Vec<u32>)> = Vec::new();
            for (node, list) in updates {
                for &nb in &list {
                    touched.push((nb as usize, vec![node as u32]));
                }
                adj[node] = list;
            }
            // 按目标节点聚合反向边
            touched.sort_unstable_by_key(|x| x.0);
            let mut reverse_groups: Vec<(usize, Vec<u32>)> = Vec::new();
            for (dst, srcs) in touched {
                if let Some(last) = reverse_groups.last_mut()
                    && last.0 == dst
                {
                    last.1.extend_from_slice(&srcs);
                } else {
                    reverse_groups.push((dst, srcs));
                }
            }
            let rev_pruned: Vec<(usize, Vec<u32>)> = reverse_groups
                .into_par_iter()
                .map(|(v, srcs)| {
                    let mut merged = adj[v].clone();
                    merged.extend_from_slice(&srcs);
                    merged.sort_unstable();
                    merged.dedup();
                    if merged.len() <= m_max {
                        return (v, merged);
                    }
                    let mut cands: Vec<(OrdDist, u32)> = merged
                        .into_iter()
                        .filter(|&x| x as usize != v)
                        .map(|x| (OrdDist(dist_fn(v, x as usize)), x))
                        .collect();
                    cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                    let pruned = prune_dispatch(v, &cands, m_max, 1.0, &dist_fn, quadri);
                    (v, pruned)
                })
                .collect();
            for (v, list) in rev_pruned {
                adj[v] = list;
            }
            cursor = end;
            batch_cap = (batch_cap * 2).min(8_192);
        }

        // Pass 2：全图成型后以 α 做一次全局并行细化 + 双向回插
        let next_lists: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|node| {
                let local_seeds = &adj[node][..adj[node].len().min(4)];
                let mut cands = beam_collect_internal(
                    &adj,
                    &entries,
                    local_seeds,
                    ef_construction,
                    n,
                    Some(node),
                    |other| dist_fn(node, other),
                );
                for &nb in &adj[node] {
                    if nb as usize != node {
                        cands.push((OrdDist(dist_fn(node, nb as usize)), nb));
                    }
                }
                cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                cands.dedup_by_key(|x| x.1);
                prune_dispatch(node, &cands, m_max, alpha, &dist_fn, quadri)
            })
            .collect();
        adj = next_lists;

        let mut reverse_buckets: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (u, nbs) in adj.iter().enumerate() {
            for &v in nbs {
                reverse_buckets[v as usize].push(u as u32);
            }
        }
        adj = (0..n)
            .into_par_iter()
            .map(|node| {
                let mut merged = adj[node].clone();
                merged.extend_from_slice(&reverse_buckets[node]);
                merged.sort_unstable();
                merged.dedup();
                if merged.len() <= m_max {
                    return merged;
                }
                let mut cands: Vec<(OrdDist, u32)> = merged
                    .into_iter()
                    .filter(|&v| v as usize != node)
                    .map(|v| (OrdDist(dist_fn(node, v as usize)), v))
                    .collect();
                cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                prune_dispatch(node, &cands, m_max, alpha, &dist_fn, quadri)
            })
            .collect();

        ensure_reachability(&mut adj, &entries, m_max, &dist_fn);

        Self {
            neighbors: adj,
            entry_points: entries,
            m_max,
        }
    }
}

/// 保证从 `entries` 出发的有向图 100% BFS 可达（等价于 NSG / DiskANN 的孤立簇连通修复，不增加额外图度数）。
fn ensure_reachability<F>(adj: &mut [Vec<u32>], entries: &[u32], m_max: usize, dist_fn: &F)
where
    F: Fn(usize, usize) -> f64,
{
    let n = adj.len();
    if n == 0 || m_max == 0 {
        return;
    }
    let mut parent = vec![usize::MAX; n];
    let mut is_entry = vec![false; n];
    let mut reached = vec![false; n];
    let mut reached_nodes = Vec::with_capacity(n);
    let mut queue = std::collections::VecDeque::new();
    for &e in entries {
        let eu = e as usize;
        if eu < n {
            is_entry[eu] = true;
            if !reached[eu] {
                reached[eu] = true;
                reached_nodes.push(eu);
                queue.push_back(eu);
            }
        }
    }
    let drain_bfs = |q: &mut std::collections::VecDeque<usize>,
                     adj_ref: &[Vec<u32>],
                     seen: &mut [bool],
                     par: &mut [usize],
                     list: &mut Vec<usize>| {
        while let Some(u) = q.pop_front() {
            for &v in &adj_ref[u] {
                let vi = v as usize;
                if !seen[vi] {
                    seen[vi] = true;
                    par[vi] = u;
                    list.push(vi);
                    q.push_back(vi);
                }
            }
        }
    };
    drain_bfs(
        &mut queue,
        adj,
        &mut reached,
        &mut parent,
        &mut reached_nodes,
    );

    let mut scan = 0usize;
    while reached_nodes.len() < n {
        while scan < n && reached[scan] {
            scan += 1;
        }
        if scan >= n {
            break;
        }
        let target = scan;
        // 优先从反向可达邻居或非入口枢纽的空闲/非生成树槽位接入，严禁覆写主入口枢纽的满边槽位
        let mut best_src: Option<(f64, usize, bool)> = None;
        for &nb in &adj[target] {
            let r = nb as usize;
            if reached[r] {
                let is_free = adj[r].len() < m_max;
                let has_slot =
                    is_free || (!is_entry[r] && adj[r].iter().any(|&v| parent[v as usize] != r));
                if has_slot {
                    let cand = (dist_fn(r, target), r, is_free);
                    if best_src.is_none_or(|b| (cand.2 && !b.2) || (cand.2 == b.2 && cand.0 < b.0))
                    {
                        best_src = Some(cand);
                    }
                }
            }
        }
        // 若反向邻接点均为满度枢纽节点，主动在可达集合里优先查找有空闲槽位且靠近中后段的节点，不破坏早期核心导航边
        if best_src.is_none_or(|b| !b.2) && !reached_nodes.is_empty() {
            let stride = (reached_nodes.len() / 256).max(1);
            let skip_core = if reached_nodes.len() > 1_024 {
                reached_nodes.len() / 4
            } else {
                0
            };
            for idx in (skip_core..reached_nodes.len()).step_by(stride) {
                let r = reached_nodes[idx];
                let is_free = adj[r].len() < m_max;
                let has_slot =
                    is_free || (!is_entry[r] && adj[r].iter().any(|&v| parent[v as usize] != r));
                if has_slot {
                    let cand = (dist_fn(r, target), r, is_free);
                    if best_src.is_none_or(|b| (cand.2 && !b.2) || (cand.2 == b.2 && cand.0 < b.0))
                    {
                        best_src = Some(cand);
                    }
                }
            }
        }
        let chosen_r = best_src.map(|x| x.1).unwrap_or_else(|| entries[0] as usize);
        if !adj[chosen_r].contains(&(target as u32)) {
            if adj[chosen_r].len() < m_max {
                adj[chosen_r].push(target as u32);
            } else if let Some(pos) = adj[chosen_r]
                .iter()
                .rposition(|&v| v as usize != target && parent[v as usize] != chosen_r)
            {
                adj[chosen_r][pos] = target as u32;
            } else {
                adj[chosen_r].push(target as u32);
            }
        }
        reached[target] = true;
        parent[target] = chosen_r;
        reached_nodes.push(target);
        queue.push_back(target);
        drain_bfs(
            &mut queue,
            adj,
            &mut reached,
            &mut parent,
            &mut reached_nodes,
        );
    }
}

fn beam_collect_internal<F>(
    adj: &[Vec<u32>],
    seeds: &[u32],
    extra_seeds: &[u32],
    ef: usize,
    n: usize,
    skip_node: Option<usize>,
    dist_fn: F,
) -> Vec<(OrdDist, u32)>
where
    F: Fn(usize) -> f64,
{
    let mut visited = vec![0u64; n.div_ceil(64)];
    if let Some(sk) = skip_node
        && sk < n
    {
        visit_check_set(&mut visited, sk);
    }
    let mut candidates: BinaryHeap<Reverse<(OrdDist, u32)>> = BinaryHeap::with_capacity(ef * 2);
    let mut results: BinaryHeap<(OrdDist, u32)> = BinaryHeap::with_capacity(ef + 1);
    for &seed in seeds.iter().chain(extra_seeds.iter()) {
        let s = seed as usize;
        if s < n && !visit_check_set(&mut visited, s) {
            let d = OrdDist(dist_fn(s));
            candidates.push(Reverse((d, seed)));
            results.push((d, seed));
            if results.len() > ef {
                results.pop();
            }
        }
    }
    while let Some(Reverse((cand_d, cur))) = candidates.pop() {
        if results.len() >= ef && cand_d > results.peek().unwrap().0 {
            break;
        }
        for &nb in &adj[cur as usize] {
            let ni = nb as usize;
            if visit_check_set(&mut visited, ni) {
                continue;
            }
            let d = OrdDist(dist_fn(ni));
            if results.len() < ef || d < results.peek().unwrap().0 {
                candidates.push(Reverse((d, nb)));
                results.push((d, nb));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }
    let mut out = results.into_vec();
    out.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    out
}

fn prune_dispatch<F>(
    target: usize,
    candidates: &[(OrdDist, u32)],
    limit: usize,
    alpha: f64,
    dist_fn: &F,
    quadri: Option<&QuadriPruneSpec<'_>>,
) -> Vec<u32>
where
    F: Fn(usize, usize) -> f64,
{
    if let Some(spec) = quadri {
        quadri_prune_candidates(target, candidates, limit, alpha, dist_fn, spec)
    } else {
        prune_candidates(target, candidates, limit, alpha * alpha, dist_fn)
    }
}

/// 广义四向黎曼罗盘剪枝（Stem-Excess 测地主干 + 等深球面环 + 双向径向升降梯 + 补边）。
fn quadri_prune_candidates<F>(
    target: usize,
    candidates: &[(OrdDist, u32)],
    limit: usize,
    alpha: f64,
    dist_fn: &F,
    spec: &QuadriPruneSpec<'_>,
) -> Vec<u32>
where
    F: Fn(usize, usize) -> f64,
{
    let geo_limit = (((limit as f64) * spec.geo_fraction).round() as usize).clamp(1, limit);
    let alpha_sq = alpha * alpha;
    let depth_t = spec.depths[target];
    let elev_dom = spec.elev_min * 0.5;
    let mut selected: Vec<(u32, f64)> = Vec::with_capacity(limit);

    // 1. 测地纤维主干配额（Stem-Excess 消去共享径向干细胞遮蔽 + 球面角单调过滤）
    for &(OrdDist(d_sq), cand) in candidates {
        let ci = cand as usize;
        if ci == target {
            continue;
        }
        if selected.len() >= geo_limit {
            break;
        }
        let d_tc = d_sq.max(0.0).sqrt();
        let r_tc = (spec.radial_dist_fn)(target, ci);
        let ang_tc = (spec.angular_loss_fn)(target, ci);
        let occluded = selected.iter().any(|&(sel, _)| {
            let si = sel as usize;
            let ang_sc = (spec.angular_loss_fn)(ci, si);
            if ang_sc > ang_tc {
                return false;
            }
            let r_sc = (spec.radial_dist_fn)(si, ci);
            let stem_excess = (r_tc - r_sc).max(0.0);
            let eff_target = (d_tc - stem_excess).max(0.0);
            if alpha * r_sc > eff_target {
                return false;
            }
            alpha_sq * dist_fn(ci, si) <= eff_target * eff_target
        });
        if !occluded {
            selected.push((cand, spec.depths[ci]));
        }
    }

    // 紧致单空间残差锥簇下已由全配额 Stem-Excess 测地主干涵盖，仅需测地保底补边；连续各向同性或多块乘积流形启用等深环与双向升降梯
    if selected.len() < limit && spec.geo_fraction < 0.999 {
        struct PoolItem {
            ang_tc: f64,
            rad_tc: f64,
            d_tc: f64,
            d_sq: f64,
            cand: u32,
            depth_c: f64,
        }
        let mut pool: Vec<PoolItem> = Vec::with_capacity(candidates.len());
        let mut min_ang = f64::INFINITY;
        for &(OrdDist(d_sq), cand) in candidates {
            let ci = cand as usize;
            if ci == target || selected.iter().any(|&(s, _)| s == cand) {
                continue;
            }
            let ang_tc = (spec.angular_loss_fn)(target, ci);
            if ang_tc < min_ang {
                min_ang = ang_tc;
            }
            pool.push(PoolItem {
                ang_tc,
                rad_tc: (spec.radial_dist_fn)(target, ci),
                d_tc: d_sq.max(0.0).sqrt(),
                d_sq,
                cand,
                depth_c: spec.depths[ci],
            });
        }
        let ang_guard = min_ang + spec.ang_guard_delta;

        // 2a. 共形/等深极限球面环（Horosphere / Iso-Shell Ring）
        let iso_local_target = if spec.bridge_cap > 0 {
            ((limit * 3) / 4).max(selected.len()).min(limit)
        } else {
            ((limit * 7) / 8).max(selected.len()).min(limit)
        };
        pool.sort_unstable_by(|a, b| {
            let (sa, sb) = if spec.iso_score_uses_geodesic {
                (a.ang_tc + 0.08 * a.d_tc, b.ang_tc + 0.08 * b.d_tc)
            } else {
                (a.ang_tc + 0.45 * a.rad_tc, b.ang_tc + 0.45 * b.rad_tc)
            };
            sa.total_cmp(&sb)
                .then_with(|| a.d_sq.total_cmp(&b.d_sq))
                .then_with(|| a.cand.cmp(&b.cand))
        });
        let is_iso_occluded = |ci: usize, ang_tc: f64, selected: &[(u32, f64)]| -> bool {
            selected.iter().any(|&(nb, _)| {
                (spec.radial_dist_fn)(nb as usize, ci) <= spec.iso_tol
                    && alpha_sq * (spec.angular_loss_fn)(ci, nb as usize) <= ang_tc
            })
        };
        for item in &pool {
            if selected.len() >= iso_local_target {
                break;
            }
            if item.ang_tc > ang_guard {
                continue;
            }
            if !is_iso_occluded(item.cand as usize, item.ang_tc, &selected) {
                selected.push((item.cand, item.depth_c));
            }
        }

        // 2b. 可选多块乘积跨聚簇角桥（严格按 bridge_cap 封顶，防孤立聚簇困死）
        if spec.bridge_cap > 0 {
            let iso_bridge_target = (selected.len() + spec.bridge_cap)
                .min(((limit * 7) / 8).max(selected.len()))
                .min(limit);
            for item in &pool {
                if selected.len() >= iso_bridge_target {
                    break;
                }
                if selected.iter().any(|&(s, _)| s == item.cand) {
                    continue;
                }
                if !is_iso_occluded(item.cand as usize, item.ang_tc, &selected) {
                    selected.push((item.cand, item.depth_c));
                }
            }
        }

        // 3. 双向径向升降梯（向内 + 向外各自半数残余名额）
        let rem = limit.saturating_sub(selected.len());
        let elev_ang_guard = min_ang + spec.elev_ang_delta;
        if rem > 0 {
            let inward_target = selected.len() + rem / 2;
            for &(is_outward, target_len) in &[(false, inward_target), (true, limit)] {
                let signed_gap = |d_c: f64| {
                    if is_outward {
                        d_c - depth_t
                    } else {
                        depth_t - d_c
                    }
                };
                pool.sort_unstable_by(|a, b| {
                    let sa = a.ang_tc + 0.15 * (signed_gap(a.depth_c) - spec.elev_step).abs();
                    let sb = b.ang_tc + 0.15 * (signed_gap(b.depth_c) - spec.elev_step).abs();
                    sa.total_cmp(&sb)
                        .then_with(|| a.d_sq.total_cmp(&b.d_sq))
                        .then_with(|| a.cand.cmp(&b.cand))
                });
                for item in &pool {
                    if selected.len() >= target_len {
                        break;
                    }
                    let valid_shell = if is_outward {
                        item.depth_c > depth_t + spec.elev_min
                    } else {
                        item.depth_c < depth_t - spec.elev_min
                    };
                    if item.ang_tc > elev_ang_guard
                        || !valid_shell
                        || selected.iter().any(|&(s, _)| s == item.cand)
                    {
                        continue;
                    }
                    let ci = item.cand as usize;
                    let occluded = selected.iter().any(|&(nb, d_nb)| {
                        let dom_shell = if is_outward {
                            d_nb > depth_t + elev_dom
                        } else {
                            d_nb < depth_t - elev_dom
                        };
                        dom_shell
                            && alpha_sq * (spec.angular_loss_fn)(ci, nb as usize) <= item.ang_tc
                    });
                    if !occluded {
                        selected.push((item.cand, item.depth_c));
                    }
                }
            }
        }
    }

    // 4. 测地近邻保底补满至 limit
    if selected.len() < limit {
        for &(OrdDist(_), cand) in candidates {
            if selected.len() >= limit {
                break;
            }
            let ci = cand as usize;
            if ci != target && !selected.iter().any(|&(s, _)| s == cand) {
                selected.push((cand, spec.depths[ci]));
            }
        }
    }

    selected.into_iter().map(|(c, _)| c).collect()
}

fn prune_candidates<F>(
    target: usize,
    candidates: &[(OrdDist, u32)],
    limit: usize,
    alpha_eff: f64,
    dist_fn: &F,
) -> Vec<u32>
where
    F: Fn(usize, usize) -> f64,
{
    let mut selected: Vec<u32> = Vec::with_capacity(limit);
    for &(OrdDist(d), cand) in candidates {
        let ci = cand as usize;
        if ci == target {
            continue;
        }
        if selected.len() >= limit {
            break;
        }
        let occluded = selected
            .iter()
            .any(|&sel| dist_fn(ci, sel as usize) * alpha_eff < d);
        if !occluded {
            selected.push(cand);
        }
    }
    if selected.len() < limit {
        for &(OrdDist(_), cand) in candidates {
            if selected.len() >= limit {
                break;
            }
            if cand as usize != target && !selected.contains(&cand) {
                selected.push(cand);
            }
        }
    }
    selected
}

struct SearchRequest<'a> {
    graph: &'a VarGraph,
    seeds: &'a [u32],
    ef: usize,
    budget: usize,
    top_k: usize,
    n: usize,
}

#[inline(always)]
fn nop_prefetch(_: usize) {}

/// 使用 L1 位图 visited + 滚动硬件预取的通用贪心束查询核心（供所有图对照复用，不计堆重排噪声）。
fn search_graph_with_rerank<F, R, P, B, PF>(
    req: SearchRequest<'_>,
    stage05_eval: Option<&R>,
    prune_lb: Option<&B>,
    prefetch: &PF,
    beam_dist: &F,
    exact_dist: &P,
) -> SearchOutcome
where
    F: Fn(usize) -> f64,
    R: Fn(usize) -> f64,
    P: Fn(usize) -> f64,
    B: Fn(usize, f64) -> bool,
    PF: Fn(usize),
{
    let SearchRequest {
        graph,
        seeds,
        ef,
        budget,
        top_k,
        n,
    } = req;
    let mut visited = vec![0u64; n.div_ceil(64)];
    let mut candidates: BinaryHeap<Reverse<(OrdDist, u32)>> = BinaryHeap::with_capacity(ef * 2);
    let mut results: BinaryHeap<(OrdDist, u32)> = BinaryHeap::with_capacity(ef + 1);
    let mut expansions = 0usize;

    for &seed in seeds {
        let s = seed as usize;
        if s < n && !visit_check_set(&mut visited, s) {
            prefetch(s);
            let d = OrdDist(beam_dist(s));
            candidates.push(Reverse((d, seed)));
            results.push((d, seed));
            if results.len() > ef {
                results.pop();
            }
        }
    }
    const PF_AHEAD: usize = 4;
    let mut unvisited = [0u32; 64];
    let mut overflow: Vec<u32> = Vec::new();
    while let Some(Reverse((cand_d, cur))) = candidates.pop() {
        if results.len() >= ef && cand_d > results.peek().unwrap().0 {
            break;
        }
        expansions += 1;
        let mut u_len = 0usize;
        overflow.clear();
        for &nb in &graph.neighbors[cur as usize] {
            let ni = nb as usize;
            if !visit_check_set(&mut visited, ni) {
                if u_len < unvisited.len() {
                    unvisited[u_len] = nb;
                    u_len += 1;
                } else {
                    overflow.push(nb);
                }
            }
        }
        for k in 0..u_len.min(PF_AHEAD) {
            prefetch(unvisited[k] as usize);
        }
        for k in 0..u_len {
            if k + PF_AHEAD < u_len {
                prefetch(unvisited[k + PF_AHEAD] as usize);
            }
            let nb = unvisited[k];
            let d = OrdDist(beam_dist(nb as usize));
            if results.len() < ef || d < results.peek().unwrap().0 {
                candidates.push(Reverse((d, nb)));
                results.push((d, nb));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
        for &nb in &overflow {
            let d = OrdDist(beam_dist(nb as usize));
            if results.len() < ef || d < results.peek().unwrap().0 {
                candidates.push(Reverse((d, nb)));
                results.push((d, nb));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    let mut pool = results.into_vec();
    if let Some(eval05) = stage05_eval {
        for item in &mut pool {
            item.0 = OrdDist(eval05(item.1 as usize));
        }
    }
    pool.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let limit = pool.len().min(budget.max(top_k));
    let mut top_heap: BinaryHeap<(OrdDist, usize)> = BinaryHeap::with_capacity(top_k + 1);
    let mut exact_reranks = 0usize;
    let mut radial_rejections = 0usize;

    for &(_, node) in pool.iter().take(limit) {
        let ni = node as usize;
        if let Some(lb_fn) = prune_lb
            && top_heap.len() >= top_k
        {
            let bound = top_heap.peek().unwrap().0.0;
            if lb_fn(ni, bound) {
                radial_rejections += 1;
                continue;
            }
        }
        let d = exact_dist(ni);
        exact_reranks += 1;
        let entry = (OrdDist(d), ni);
        if top_heap.len() < top_k {
            top_heap.push(entry);
        } else if entry < *top_heap.peek().unwrap() {
            top_heap.pop();
            top_heap.push(entry);
        }
    }
    let mut out = top_heap.into_vec();
    out.sort_unstable();
    SearchOutcome {
        results: out.into_iter().map(|(OrdDist(d), id)| (id, d)).collect(),
        exact_reranks,
        radial_rejections,
        expansions,
    }
}

/// 64 字节缓存行对齐基元，用于零填充合并节点元信息与 2-bit SM 签名。
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct AlignedLine([u64; 8]);

/// 球面残差锥字典（与主路径 `ResidualConeDict` 同原理，完全不依赖曲率剖面或子块度量）。
struct WarpedConeDict {
    k: usize,
    dim: usize,
    axes: Vec<f32>,
    axis_dots: Vec<f32>,
}

struct ActiveConeQuery {
    alpha: f64,
    beta: f64,
    sig: Bq2Signature,
    sm: SmCodeParams,
    asym: AsymQuerySm,
}

struct ConeFitResult {
    dict: WarpedConeDict,
    metas: Vec<(u32, f32, f32)>,
    res_vecs: Vec<f32>,
}

enum QueryDirState {
    Cone {
        dots: Vec<f64>,
        active: Vec<Option<ActiveConeQuery>>,
    },
    Flat {
        sig: Bq2Signature,
        sm: SmCodeParams,
        asym: AsymQuerySm,
    },
}

impl QueryDirState {
    fn from_vector(raw_slice: &[f32], radius: f64, dict: Option<&WarpedConeDict>) -> Self {
        let dim = raw_slice.len();
        let inv_r = 1.0 / radius.max(1e-12);
        let q_unit: Vec<f32> = raw_slice
            .iter()
            .map(|&x| (f64::from(x) * inv_r) as f32)
            .collect();
        if let Some(d) = dict {
            let mut dots = Vec::with_capacity(d.k);
            let mut active = Vec::with_capacity(d.k);
            let mut best_c = 0usize;
            let mut best_d = f64::NEG_INFINITY;
            for c in 0..d.k {
                let ax = &d.axes[c * d.dim..(c + 1) * d.dim];
                let dot = dot_f32_f64(&q_unit, ax).clamp(-1.0, 1.0);
                if dot > best_d {
                    best_d = dot;
                    best_c = c;
                }
                dots.push(dot);
                active.push(None);
            }
            let mut q_res = vec![0.0f32; dim];
            for c in 0..d.k {
                if dots[c] >= 0.45 || c == best_c {
                    let a_q = dots[c];
                    let b_q = (1.0 - a_q * a_q).max(0.0).sqrt();
                    let ax = &d.axes[c * d.dim..(c + 1) * d.dim];
                    let mut sum_sq = 0.0f64;
                    for ((dst, &qu), &au) in q_res.iter_mut().zip(&q_unit).zip(ax) {
                        let v = f64::from(qu) - a_q * f64::from(au);
                        sum_sq += v * v;
                        *dst = v as f32;
                    }
                    let r_norm = sum_sq.sqrt();
                    if r_norm > 1e-6 {
                        let inv_rn = 1.0 / r_norm;
                        for x in &mut q_res {
                            *x = (f64::from(*x) * inv_rn) as f32;
                        }
                    } else {
                        q_res.copy_from_slice(ax);
                    }
                    let sig = Bq2Signature::from_vector(&q_res);
                    let (sm, _) = SmCodeParams::from_vector_and_sig(&q_res, &sig);
                    let mut asym = AsymQuerySm::new();
                    asym.encode_into(&q_res, 1.0);
                    active[c] = Some(ActiveConeQuery {
                        alpha: a_q,
                        beta: b_q,
                        sig,
                        sm,
                        asym,
                    });
                }
            }
            Self::Cone { dots, active }
        } else {
            let sig = Bq2Signature::from_vector(&q_unit);
            let (sm, _) = SmCodeParams::from_vector_and_sig(&q_unit, &sig);
            let mut asym = AsymQuerySm::new();
            asym.encode_into(&q_unit, 1.0);
            Self::Flat { sig, sm, asym }
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn eval_beam_cos(
        &self,
        cone_id: usize,
        m_alpha: f64,
        m_beta: f64,
        m_sm: SmCodeParams,
        pos_ptr: *const u64,
        strong_ptr: *const u64,
        dim: usize,
    ) -> f64 {
        match self {
            Self::Cone { dots, active } => {
                if let Some(aq) = &active[cone_id] {
                    let cos_res = if m_beta <= 1e-6 || aq.beta <= 1e-6 {
                        0.0
                    } else {
                        let counts = sm_dot_counts_raw(
                            pos_ptr,
                            strong_ptr,
                            aq.sig.pos.as_ptr(),
                            aq.sig.strong.as_ptr(),
                            dim,
                        );
                        sm_calibrated_cosine_fast(aq.sm, m_sm, counts, dim)
                    };
                    (aq.alpha * m_alpha + aq.beta * m_beta * cos_res).clamp(-1.0, 1.0)
                } else {
                    (m_alpha * dots[cone_id]).clamp(-1.0, 1.0)
                }
            }
            Self::Flat { sig, sm, .. } => {
                let counts = sm_dot_counts_raw(
                    pos_ptr,
                    strong_ptr,
                    sig.pos.as_ptr(),
                    sig.strong.as_ptr(),
                    dim,
                );
                sm_calibrated_cosine_fast(*sm, m_sm, counts, dim)
            }
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn eval_stage05_cos(
        &self,
        cone_id: usize,
        m_alpha: f64,
        m_beta: f64,
        m_sm: SmCodeParams,
        pos_slice: &[u64],
        strong_slice: &[u64],
    ) -> f64 {
        match self {
            Self::Cone { dots, active } => {
                if let Some(aq) = &active[cone_id] {
                    let cos_res = if m_beta <= 1e-6 || aq.beta <= 1e-6 {
                        0.0
                    } else {
                        aq.asym
                            .calibrated_cosine_slices(pos_slice, strong_slice, m_sm)
                    };
                    (aq.alpha * m_alpha + aq.beta * m_beta * cos_res).clamp(-1.0, 1.0)
                } else {
                    (m_alpha * dots[cone_id]).clamp(-1.0, 1.0)
                }
            }
            Self::Flat { asym, .. } => asym.calibrated_cosine_slices(pos_slice, strong_slice, m_sm),
        }
    }

    /// 无任何反三角调用的确定性球面 Cauchy-Schwarz 最大余弦上界（`<= 1.0`），用于无冷存访问的几何保真剪枝。
    #[inline(always)]
    fn cos_upper_bound(&self, cone_id: usize, m_alpha: f64, m_beta: f64) -> Option<f64> {
        match self {
            Self::Cone { dots, .. } => {
                let a_q = dots[cone_id];
                let b_q = (1.0 - a_q * a_q).max(0.0).sqrt();
                Some((a_q * m_alpha + b_q * m_beta + 5e-4).clamp(-1.0, 1.0))
            }
            Self::Flat { .. } => None,
        }
    }
}

impl WarpedConeDict {
    /// 自适应从单位方向集合中学习球面残差锥中心并做球面残差分解（零单点堆分配并行编码）。
    /// 若聚簇对齐比例不足 50%（例如各向同性均匀分布），返回 `None` 并安全回退到裸 2-bit SM。
    fn fit_and_encode(vectors: &[f32], dim: usize, n: usize) -> Option<ConeFitResult> {
        if n == 0 || dim == 0 {
            return None;
        }
        let sample_n = n.min(4_096);
        let mut s_rng = DetRng::new(0xC0DE_5EED);
        let mut samples: Vec<Vec<f32>> = Vec::with_capacity(sample_n);
        for s_idx in 0..sample_n {
            let i = if sample_n == n {
                s_idx
            } else {
                (s_rng.next_u64() as usize) % n
            };
            let row = &vectors[i * dim..(i + 1) * dim];
            let norm = norm_squared_f64(row).sqrt().max(1e-12);
            samples.push(row.iter().map(|&x| (f64::from(x) / norm) as f32).collect());
        }

        let max_cones = 64.min(samples.len());
        let mut centers: Vec<Vec<f32>> = Vec::with_capacity(max_cones);
        centers.push(samples[0].clone());
        let mut max_sim = vec![f64::NEG_INFINITY; samples.len()];
        while centers.len() < max_cones {
            let last = centers.last().unwrap();
            let mut min_best = f64::INFINITY;
            let mut far_idx = 0usize;
            for (si, s) in samples.iter().enumerate() {
                let d = dot_f32_f64(s, last);
                if d > max_sim[si] {
                    max_sim[si] = d;
                }
                if max_sim[si] < min_best {
                    min_best = max_sim[si];
                    far_idx = si;
                }
            }
            if min_best >= 0.70 {
                break;
            }
            centers.push(samples[far_idx].clone());
        }

        let k = centers.len();
        for _ in 0..4 {
            let mut sums = vec![0.0f64; k * dim];
            let mut counts = vec![0usize; k];
            for s in &samples {
                let mut best_c = 0usize;
                let mut best_d = f64::NEG_INFINITY;
                for (ci, c) in centers.iter().enumerate() {
                    let d = dot_f32_f64(s, c);
                    if d > best_d {
                        best_d = d;
                        best_c = ci;
                    }
                }
                counts[best_c] += 1;
                let off = best_c * dim;
                for (j, &val) in s.iter().enumerate() {
                    sums[off + j] += f64::from(val);
                }
            }
            for (ci, cnt) in counts.iter().enumerate() {
                if *cnt > 0 {
                    let off = ci * dim;
                    let norm = sums[off..off + dim]
                        .iter()
                        .map(|&v| v * v)
                        .sum::<f64>()
                        .sqrt()
                        .max(1e-12);
                    for j in 0..dim {
                        centers[ci][j] = (sums[off + j] / norm) as f32;
                    }
                }
            }
        }

        let mut aligned = 0usize;
        for s in &samples {
            let best_d = centers
                .iter()
                .map(|c| dot_f32_f64(s, c))
                .fold(f64::NEG_INFINITY, f64::max);
            if best_d >= 0.55 {
                aligned += 1;
            }
        }
        if aligned * 2 < samples.len() {
            return None;
        }

        let mut axes = Vec::with_capacity(k * dim);
        for c in &centers {
            axes.extend_from_slice(c);
        }
        let mut axis_dots = vec![0.0f32; k * k];
        for i in 0..k {
            for j in i..k {
                let d = dot_f32_f64(&axes[i * dim..(i + 1) * dim], &axes[j * dim..(j + 1) * dim])
                    .clamp(-1.0, 1.0) as f32;
                axis_dots[i * k + j] = d;
                axis_dots[j * k + i] = d;
            }
        }

        let mut cone_meta = vec![(0u32, 0.0f32, 1.0f32); n];
        let mut res_vecs = vec![0.0f32; n * dim];
        res_vecs
            .par_chunks_exact_mut(dim)
            .zip(cone_meta.par_iter_mut())
            .enumerate()
            .for_each_init(
                || vec![0.0f32; dim],
                |u_buf, (i, (res_out, meta_out))| {
                    let row = &vectors[i * dim..(i + 1) * dim];
                    let norm = norm_squared_f64(row).sqrt().max(1e-12);
                    let inv_norm = 1.0 / norm;
                    for (dst, &x) in u_buf.iter_mut().zip(row) {
                        *dst = (f64::from(x) * inv_norm) as f32;
                    }
                    let mut best_c = 0usize;
                    let mut best_a = f64::NEG_INFINITY;
                    for ci in 0..k {
                        let a = dot_f32_f64(u_buf, &axes[ci * dim..(ci + 1) * dim]);
                        if a > best_a {
                            best_a = a;
                            best_c = ci;
                        }
                    }
                    let alpha = best_a.clamp(-1.0, 1.0);
                    let beta = (1.0 - alpha * alpha).max(0.0).sqrt();
                    let ax = &axes[best_c * dim..(best_c + 1) * dim];
                    let mut r_sq = 0.0f64;
                    for ((dst, &ui), &ai) in res_out.iter_mut().zip(u_buf.iter()).zip(ax) {
                        let v = f64::from(ui) - alpha * f64::from(ai);
                        r_sq += v * v;
                        *dst = v as f32;
                    }
                    let r_norm = r_sq.sqrt();
                    if r_norm > 1e-6 {
                        let inv_rn = 1.0 / r_norm;
                        for x in res_out.iter_mut() {
                            *x = (f64::from(*x) * inv_rn) as f32;
                        }
                    } else {
                        res_out.copy_from_slice(ax);
                    }
                    *meta_out = (best_c as u32, alpha as f32, beta as f32);
                },
            );

        Some(ConeFitResult {
            dict: Self {
                k,
                dim,
                axes,
                axis_dots,
            },
            metas: cone_meta,
            res_vecs,
        })
    }

    #[inline(always)]
    fn axis_dot(&self, a: u32, b: u32) -> f64 {
        f64::from(self.axis_dots[(a as usize) * self.k + (b as usize)])
    }

    fn memory_bytes(&self) -> usize {
        (self.axes.len() + self.axis_dots.len()) * 4
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  层级 2：混合/乘积常曲率流形（∏ H_{-c_j}^{d_j} × R^{d_0} × ∏ S_{κ_k}^{d_k}）
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub enum ProductFactor {
    /// 曲率为 -curvature (< 0) 的 Poincaré 球子空间
    Hyperbolic { dim: usize, curvature: f64 },
    /// 欧氏平坦子空间
    Euclidean { dim: usize },
    /// 曲率为 +curvature (> 0) 的球面子空间（点位于半径 1/sqrt(curvature) 的球面上）
    Spherical { dim: usize, curvature: f64 },
}

impl ProductFactor {
    #[inline]
    pub fn dim(self) -> usize {
        match self {
            Self::Hyperbolic { dim, .. }
            | Self::Euclidean { dim }
            | Self::Spherical { dim, .. } => dim,
        }
    }
}

/// 精确计算混合/乘积流形平方测地距离 d_prod^2(a, b) = ∑_j d_j^2。
pub fn exact_product_distance_sq(factors: &[ProductFactor], a: &[f32], b: &[f32]) -> f64 {
    let mut off = 0usize;
    let mut sum_sq = 0.0f64;
    for &f in factors {
        let d = f.dim();
        let sa = &a[off..off + d];
        let sb = &b[off..off + d];
        off += d;
        match f {
            ProductFactor::Hyperbolic { curvature, .. } => {
                let na = norm_squared_f64(sa);
                let nb = norm_squared_f64(sb);
                let fa = (1.0 - curvature * na).max(1e-15).recip();
                let fb = (1.0 - curvature * nb).max(1e-15).recip();
                let proxy = poincare_proxy(sa, sb, fa * fb);
                let dist = poincare_distance(proxy, curvature);
                sum_sq += dist * dist;
            }
            ProductFactor::Euclidean { .. } => {
                sum_sq += poincare_proxy(sa, sb, 1.0);
            }
            ProductFactor::Spherical { curvature, .. } => {
                // 对球面因子，两点模长理论上均为 1/sqrt(κ)，为防浮点微差先单位化后求测地角。
                let r_prod = (norm_squared_f64(sa) * norm_squared_f64(sb)).sqrt();
                let cos_t = if r_prod > 1e-18 {
                    (dot_f32_f64(sa, sb) / r_prod).clamp(-1.0, 1.0)
                } else {
                    1.0
                };
                let theta = cos_t.acos();
                sum_sq += (theta * theta) / curvature;
            }
        }
    }
    sum_sq
}

const FLAT_CONE_ID: u8 = u8::MAX;

/// 分块极坐标节点元数据（严格 32B/子空间块，保留径向/共形标量 + 块内 SM 重构参数 + 零开销打包残差锥索引）。
#[repr(C, align(32))]
#[derive(Clone, Copy)]
struct BlockNodeMeta {
    radius: f32,
    boundary_factor: f32,
    depth: f32,
    sm_weak: f32,
    sm_delta: f32,
    sm_sigma: f32,
    strong_ones: u32,
    cone_packed: u32,
}

impl BlockNodeMeta {
    #[inline(always)]
    fn pack_cone(cone_id: u8, alpha: f32) -> u32 {
        let q15 = (alpha * 32767.0).round().clamp(-32767.0, 32767.0) as i16;
        u32::from(cone_id) | (u32::from(q15 as u16) << 16)
    }

    fn from_block_with_cone(
        raw_slice: &[f32],
        encoded_slice: &[f32],
        factor: ProductFactor,
        store: &Bq2Store,
        idx: usize,
        cone_info: Option<(u32, f32, f32)>,
    ) -> Self {
        let radius = norm_squared_f64(raw_slice).sqrt();
        let (sm, _) = SmCodeParams::from_vector_and_store(encoded_slice, store, idx);
        let (boundary_factor, depth) = match factor {
            ProductFactor::Hyperbolic { curvature, .. } => {
                let r_sq = radius * radius;
                let bf = (1.0 - curvature * r_sq).max(1e-12).recip();
                let s = curvature.sqrt();
                let d = 2.0 * (s * radius).clamp(0.0, 1.0 - 1e-12).atanh() / s;
                (bf as f32, d as f32)
            }
            ProductFactor::Euclidean { .. } => (1.0, radius as f32),
            ProductFactor::Spherical { .. } => (1.0, 0.0),
        };
        let cone_packed = match cone_info {
            Some((cid, alpha, _)) => Self::pack_cone(cid.min(254) as u8, alpha),
            None => Self::pack_cone(FLAT_CONE_ID, 0.0),
        };
        Self {
            radius: radius as f32,
            boundary_factor,
            depth,
            sm_weak: sm.weak,
            sm_delta: sm.delta,
            sm_sigma: sm.sigma,
            strong_ones: sm.strong_ones,
            cone_packed,
        }
    }

    #[inline(always)]
    fn sm_params(self) -> SmCodeParams {
        SmCodeParams {
            weak: self.sm_weak,
            delta: self.sm_delta,
            sigma: self.sm_sigma,
            strong_ones: self.strong_ones,
        }
    }

    #[inline(always)]
    fn cone_id(self) -> usize {
        (self.cone_packed & 0xFF) as usize
    }

    #[inline(always)]
    fn cone_alpha(self) -> f64 {
        let q15 = (self.cone_packed >> 16) as u16 as i16;
        f64::from(q15) * (1.0 / 32767.0)
    }

    #[inline(always)]
    fn cone_beta(self) -> f64 {
        let a = self.cone_alpha();
        (1.0 - a * a).max(0.0).sqrt()
    }
}

/// 单条查询在乘积流形分块表示上的预编码状态（支持每块自适应残差锥勾股加速，每查询仅构建一次）。
struct BlockQuery {
    dir: QueryDirState,
    radius: f64,
    boundary_factor: f64,
    depth: f64,
}

impl BlockQuery {
    fn new(slice: &[f32], factor: ProductFactor, dict: Option<&WarpedConeDict>) -> Self {
        let radius = norm_squared_f64(slice).sqrt();
        let dir = QueryDirState::from_vector(slice, radius, dict);
        let (boundary_factor, depth) = match factor {
            ProductFactor::Hyperbolic { curvature, .. } => {
                let s = curvature.sqrt();
                let bf = (1.0 - curvature * radius * radius).max(1e-12).recip();
                let d = 2.0 * (s * radius).clamp(0.0, 1.0 - 1e-12).atanh() / s;
                (bf, d)
            }
            ProductFactor::Euclidean { .. } => (1.0, radius),
            ProductFactor::Spherical { .. } => (1.0, 0.0),
        };
        Self {
            dir,
            radius,
            boundary_factor,
            depth,
        }
    }
}

#[inline(always)]
fn fast_ln_pos(v: f64) -> f64 {
    let bits = v.to_bits();
    let exp = ((bits >> 52) as i64) - 1023;
    let mant = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000);
    let y = (mant - 1.0) / (mant + 1.0);
    let y2 = y * y;
    let ln_m = 2.0 * y * (1.0 + y2 * (1.0 / 3.0 + y2 * (0.2 + y2 * (1.0 / 7.0 + y2 / 9.0))));
    (exp as f64) * std::f64::consts::LN_2 + ln_m
}

#[inline]
fn fast_block_dist_sq(
    factor: ProductFactor,
    qr: f64,
    qf: f64,
    vr: f64,
    vf: f64,
    cos_hat: f64,
    fast_surrogate: bool,
) -> f64 {
    match factor {
        ProductFactor::Hyperbolic { curvature, .. } => {
            let dr = qr - vr;
            let angular = (qr * vr * (1.0 - cos_hat)).max(0.0);
            let proxy = (dr * dr + 2.0 * angular) * qf * vf;
            if fast_surrogate {
                // 纯内联 IEEE-754 对数逼近取代 libm::ln，消除束搜索热路径函数调用开销
                let x = 2.0 * curvature * proxy;
                let a = if x < 1e-4 {
                    (2.0 * x).sqrt() * (1.0 - x / 12.0)
                } else {
                    fast_ln_pos(1.0 + x + (x * (x + 2.0)).sqrt())
                };
                (a * a) / curvature
            } else {
                let d = poincare_distance(proxy, curvature);
                d * d
            }
        }
        ProductFactor::Euclidean { .. } => {
            let dr = qr - vr;
            (dr * dr + 2.0 * qr * vr * (1.0 - cos_hat)).max(0.0)
        }
        ProductFactor::Spherical { curvature, .. } => {
            if fast_surrogate {
                // θ^2 ≈ 2(1-c) * (1 + (1-c)/6 + 4*(1-c)^2/90) 在 [0, π/2] 内误差 < 1% 且零反三角调用
                let t = (1.0 - cos_hat).clamp(0.0, 2.0);
                let theta2 = 2.0 * t * (1.0 + t * (1.0 / 6.0 + t * (4.0 / 90.0)));
                theta2 / curvature
            } else {
                let theta = cos_hat.clamp(-1.0, 1.0).acos();
                (theta * theta) / curvature
            }
        }
    }
}

/// 层级 2 混合/乘积流形分块极坐标 SM 索引（单节点 64B 缓存行合表 + 块级残差锥字典 + 四向黎曼罗盘）。
pub struct ProductBlockSmIndex {
    pub factors: Vec<ProductFactor>,
    pub total_dim: usize,
    pub vectors: Vec<f32>,
    pub n: usize,
    block_offsets: Vec<usize>,
    block_chunks: Vec<usize>,
    sig_byte_offsets: Vec<usize>,
    node_stride_bytes: usize,
    hot_slab: Vec<AlignedLine>,
    block_cones: Vec<Option<WarpedConeDict>>,
    pub graph: VarGraph,
}

impl ProductBlockSmIndex {
    #[inline(always)]
    fn hot_slab_ptr(&self, node: usize) -> *const u8 {
        unsafe { (self.hot_slab.as_ptr() as *const u8).add(node * self.node_stride_bytes) }
    }

    #[inline(always)]
    fn block_meta(&self, node: usize, bi: usize) -> BlockNodeMeta {
        unsafe { *(self.hot_slab_ptr(node).cast::<BlockNodeMeta>().add(bi)) }
    }

    #[inline(always)]
    fn block_sig_ptrs(&self, node: usize, bi: usize) -> (*const u64, *const u64) {
        unsafe {
            let p = self
                .hot_slab_ptr(node)
                .add(self.sig_byte_offsets[bi])
                .cast::<u64>();
            (p, p.add(self.block_chunks[bi]))
        }
    }

    #[inline(always)]
    fn block_sig_slices(&self, node: usize, bi: usize) -> (&[u64], &[u64]) {
        let chunks = self.block_chunks[bi];
        let (pp, sp) = self.block_sig_ptrs(node, bi);
        unsafe {
            (
                std::slice::from_raw_parts(pp, chunks),
                std::slice::from_raw_parts(sp, chunks),
            )
        }
    }

    #[inline(always)]
    fn prefetch_node(&self, node: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let p = self.hot_slab_ptr(node);
            let mut off = 0usize;
            while off < self.node_stride_bytes {
                _mm_prefetch(p.add(off) as *const i8, _MM_HINT_T0);
                off += 64;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = node;
    }

    pub fn build(
        factors: Vec<ProductFactor>,
        vectors: &[f32],
        m_max: usize,
        ef_construction: usize,
        alpha: f64,
        use_compass: bool,
    ) -> Self {
        let total_dim: usize = factors.iter().map(|f| f.dim()).sum();
        assert_eq!(vectors.len() % total_dim, 0);
        let n = vectors.len() / total_dim;
        let nb = factors.len();
        let mut block_offsets = Vec::with_capacity(nb);
        let mut block_chunks = Vec::with_capacity(nb);
        let mut sig_byte_offsets = Vec::with_capacity(nb);
        let mut off = 0usize;
        let mut cur_sig_off = nb * std::mem::size_of::<BlockNodeMeta>();
        for &f in &factors {
            let d = f.dim();
            let ch = d.div_ceil(64);
            block_offsets.push(off);
            block_chunks.push(ch);
            sig_byte_offsets.push(cur_sig_off);
            off += d;
            cur_sig_off += ch * 2 * 8;
        }
        let node_stride_lines = cur_sig_off.div_ceil(64);
        let node_stride_bytes = node_stride_lines * 64;

        let mut hot_slab = vec![AlignedLine([0; 8]); n * node_stride_lines];
        let mut block_cones = Vec::with_capacity(nb);

        for (bi, &f) in factors.iter().enumerate() {
            let b_off = block_offsets[bi];
            let b_dim = f.dim();
            let ch = block_chunks[bi];
            let mut raw_buf = Vec::with_capacity(n * b_dim);
            for row in vectors.chunks_exact(total_dim) {
                raw_buf.extend_from_slice(&row[b_off..b_off + b_dim]);
            }
            let fit = WarpedConeDict::fit_and_encode(&raw_buf, b_dim, n);
            let (dict_opt, cone_metas, res_opt) = if let Some(res) = fit {
                (Some(res.dict), Some(res.metas), Some(res.res_vecs))
            } else {
                (None, None, None)
            };
            let enc_src = res_opt.as_deref().unwrap_or(&raw_buf);
            let store = Bq2Store::from_vectors_parallel(enc_src, b_dim);

            let block_metas: Vec<BlockNodeMeta> = (0..n)
                .into_par_iter()
                .map(|node| {
                    let raw_slice = &raw_buf[node * b_dim..(node + 1) * b_dim];
                    let enc_slice = &enc_src[node * b_dim..(node + 1) * b_dim];
                    let c_info = cone_metas.as_ref().map(|cm| cm[node]);
                    BlockNodeMeta::from_block_with_cone(
                        raw_slice, enc_slice, f, &store, node, c_info,
                    )
                })
                .collect();

            // 写入单节点合表连续内存（先元数据块，再紧随各块 pos/strong 位平面）
            let slab_base = hot_slab.as_mut_ptr() as *mut u8;
            let sig_off = sig_byte_offsets[bi];
            for node in 0..n {
                unsafe {
                    let base = slab_base.add(node * node_stride_bytes);
                    *base.cast::<BlockNodeMeta>().add(bi) = block_metas[node];
                    let dst_pos = base.add(sig_off).cast::<u64>();
                    let dst_strong = dst_pos.add(ch);
                    std::ptr::copy_nonoverlapping(store.pos_slice(node).as_ptr(), dst_pos, ch);
                    std::ptr::copy_nonoverlapping(
                        store.strong_slice(node).as_ptr(),
                        dst_strong,
                        ch,
                    );
                }
            }
            block_cones.push(dict_opt);
        }

        let mut tmp_idx = Self {
            factors,
            total_dim,
            vectors: Vec::new(),
            n,
            block_offsets,
            block_chunks,
            sig_byte_offsets,
            node_stride_bytes,
            hot_slab,
            block_cones,
            graph: VarGraph {
                neighbors: Vec::new(),
                entry_points: Vec::new(),
                m_max,
            },
        };

        let h_depths: Vec<f64> = (0..n)
            .map(|i| {
                (0..nb)
                    .filter(|&bi| matches!(tmp_idx.factors[bi], ProductFactor::Hyperbolic { .. }))
                    .map(|bi| f64::from(tmp_idx.block_meta(i, bi).depth))
                    .sum()
            })
            .collect();
        let entries = pick_diverse_hubs(vectors, total_dim, &h_depths, 8);

        let pair_block_cos = |left: usize, right: usize, bi: usize| -> f64 {
            let ml = tmp_idx.block_meta(left, bi);
            let mr = tmp_idx.block_meta(right, bi);
            let b_dim = tmp_idx.factors[bi].dim();
            if let Some(ref dict) = tmp_idx.block_cones[bi] {
                let cid_l = ml.cone_id();
                let cid_r = mr.cone_id();
                let al = ml.cone_alpha();
                let ar = mr.cone_alpha();
                if cid_l == cid_r {
                    let bl = (1.0 - al * al).max(0.0).sqrt();
                    let br = (1.0 - ar * ar).max(0.0).sqrt();
                    let cos_res = if bl <= 1e-6 || br <= 1e-6 {
                        0.0
                    } else {
                        let (pl, sl) = tmp_idx.block_sig_ptrs(left, bi);
                        let (pr, sr) = tmp_idx.block_sig_ptrs(right, bi);
                        let counts = sm_dot_counts_raw(pl, sl, pr, sr, b_dim);
                        sm_calibrated_cosine_fast(ml.sm_params(), mr.sm_params(), counts, b_dim)
                    };
                    (al * ar + bl * br * cos_res).clamp(-1.0, 1.0)
                } else {
                    (al * ar * dict.axis_dot(cid_l as u32, cid_r as u32)).clamp(-1.0, 1.0)
                }
            } else {
                let (pl, sl) = tmp_idx.block_sig_ptrs(left, bi);
                let (pr, sr) = tmp_idx.block_sig_ptrs(right, bi);
                let counts = sm_dot_counts_raw(pl, sl, pr, sr, b_dim);
                sm_calibrated_cosine_fast(ml.sm_params(), mr.sm_params(), counts, b_dim)
            }
        };

        let pair_sm_dist = |left: usize, right: usize| -> f64 {
            let mut sum = 0.0f64;
            for bi in 0..nb {
                let ml = tmp_idx.block_meta(left, bi);
                let mr = tmp_idx.block_meta(right, bi);
                let cos_hat = pair_block_cos(left, right, bi);
                sum += fast_block_dist_sq(
                    tmp_idx.factors[bi],
                    f64::from(ml.radius),
                    f64::from(ml.boundary_factor),
                    f64::from(mr.radius),
                    f64::from(mr.boundary_factor),
                    cos_hat,
                    true,
                );
            }
            sum
        };

        let radial_dist_fn = |u: usize, v: usize| -> f64 {
            let mut sq = 0.0f64;
            for bi in 0..nb {
                let d = f64::from(tmp_idx.block_meta(u, bi).depth)
                    - f64::from(tmp_idx.block_meta(v, bi).depth);
                sq += d * d;
            }
            sq.sqrt()
        };

        let angular_loss_fn = |u: usize, v: usize| -> f64 {
            let mut loss = 0.0f64;
            for bi in 0..nb {
                loss += 1.0 - pair_block_cos(u, v, bi);
            }
            loss / (nb as f64)
        };

        let quadri_spec = QuadriPruneSpec {
            depths: &h_depths,
            radial_dist_fn: &radial_dist_fn,
            angular_loss_fn: &angular_loss_fn,
            iso_tol: 1.2,
            elev_min: 0.25,
            elev_step: 0.85,
            geo_fraction: 0.50,
            ang_guard_delta: 0.50,
            elev_ang_delta: 0.80,
            iso_score_uses_geodesic: true,
            bridge_cap: 2,
        };

        let graph = VarGraph::build_with_quadri(
            n,
            m_max,
            ef_construction,
            alpha,
            entries,
            pair_sm_dist,
            if use_compass {
                Some(&quadri_spec)
            } else {
                None
            },
        );

        tmp_idx.vectors = vectors.to_vec();
        tmp_idx.graph = graph;
        tmp_idx
    }

    pub fn index_memory_bytes(&self) -> usize {
        let slab_bytes = self.hot_slab.len() * std::mem::size_of::<AlignedLine>();
        let cone_bytes: usize = self
            .block_cones
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|c| c.memory_bytes())
            .sum();
        let graph_bytes: usize = self.graph.neighbors.iter().map(|v| v.len() * 4).sum();
        slab_bytes + cone_bytes + graph_bytes
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
        budget: usize,
        use_radial_bound: bool,
        use_fast_surrogate: bool,
    ) -> SearchOutcome {
        let nb = self.factors.len();
        let q_blocks: Vec<BlockQuery> = (0..nb)
            .map(|bi| {
                let off = self.block_offsets[bi];
                let d = self.factors[bi].dim();
                BlockQuery::new(
                    &query[off..off + d],
                    self.factors[bi],
                    self.block_cones[bi].as_ref(),
                )
            })
            .collect();

        let beam_dist = |node: usize| -> f64 {
            let mut sum = 0.0f64;
            for bi in 0..nb {
                let m = self.block_meta(node, bi);
                let qb = &q_blocks[bi];
                let d_bi = self.factors[bi].dim();
                let (pp, sp) = self.block_sig_ptrs(node, bi);
                let cos_hat = qb.dir.eval_beam_cos(
                    m.cone_id(),
                    m.cone_alpha(),
                    m.cone_beta(),
                    m.sm_params(),
                    pp,
                    sp,
                    d_bi,
                );
                sum += fast_block_dist_sq(
                    self.factors[bi],
                    qb.radius,
                    qb.boundary_factor,
                    f64::from(m.radius),
                    f64::from(m.boundary_factor),
                    cos_hat,
                    use_fast_surrogate,
                );
            }
            sum
        };

        let stage05 = |node: usize| -> f64 {
            let mut sum = 0.0f64;
            for bi in 0..nb {
                let m = self.block_meta(node, bi);
                let qb = &q_blocks[bi];
                let (ps, ss) = self.block_sig_slices(node, bi);
                let cos_hat = qb.dir.eval_stage05_cos(
                    m.cone_id(),
                    m.cone_alpha(),
                    m.cone_beta(),
                    m.sm_params(),
                    ps,
                    ss,
                );
                sum += fast_block_dist_sq(
                    self.factors[bi],
                    qb.radius,
                    qb.boundary_factor,
                    f64::from(m.radius),
                    f64::from(m.boundary_factor),
                    cos_hat,
                    false,
                );
            }
            sum
        };

        let prune_lb = |node: usize, bound: f64| -> bool {
            let mut rad_lb = 0.0f64;
            for bi in 0..nb {
                let m = self.block_meta(node, bi);
                match self.factors[bi] {
                    ProductFactor::Hyperbolic { .. } | ProductFactor::Euclidean { .. } => {
                        let safe =
                            ((q_blocks[bi].depth - f64::from(m.depth)).abs() - 1e-4).max(0.0);
                        rad_lb += safe * safe;
                    }
                    ProductFactor::Spherical { .. } => {}
                }
            }
            if rad_lb > bound {
                return true;
            }
            // 若具备残差锥信息，追加零反三角的 Cauchy-Schwarz 严格几何角度下界
            let mut tight_lb = 0.0f64;
            for bi in 0..nb {
                let m = self.block_meta(node, bi);
                let qb = &q_blocks[bi];
                if let Some(cos_max) =
                    qb.dir
                        .cos_upper_bound(m.cone_id(), m.cone_alpha(), m.cone_beta())
                {
                    let b_lb = match self.factors[bi] {
                        ProductFactor::Hyperbolic { .. } => {
                            let d = fast_block_dist_sq(
                                self.factors[bi],
                                qb.radius,
                                qb.boundary_factor,
                                f64::from(m.radius),
                                f64::from(m.boundary_factor),
                                cos_max,
                                false,
                            );
                            (d - 1e-4).max(0.0)
                        }
                        ProductFactor::Euclidean { .. } => {
                            let d = fast_block_dist_sq(
                                self.factors[bi],
                                qb.radius,
                                qb.boundary_factor,
                                f64::from(m.radius),
                                f64::from(m.boundary_factor),
                                cos_max,
                                true,
                            );
                            (d - 1e-4).max(0.0)
                        }
                        // θ^2 >= 2(1 - cos θ) 严格成立，零 acos 即可得球面真下界
                        ProductFactor::Spherical { curvature, .. } => {
                            (2.0 * (1.0 - cos_max) / curvature - 1e-4).max(0.0)
                        }
                    };
                    tight_lb += b_lb;
                } else {
                    match self.factors[bi] {
                        ProductFactor::Hyperbolic { .. } | ProductFactor::Euclidean { .. } => {
                            let safe = ((qb.depth - f64::from(m.depth)).abs() - 1e-4).max(0.0);
                            tight_lb += safe * safe;
                        }
                        ProductFactor::Spherical { .. } => {}
                    }
                }
            }
            tight_lb > bound
        };

        let exact_dist = |node: usize| -> f64 {
            let row = &self.vectors[node * self.total_dim..(node + 1) * self.total_dim];
            exact_product_distance_sq(&self.factors, query, row)
        };
        let prefetch = |node: usize| self.prefetch_node(node);

        search_graph_with_rerank(
            SearchRequest {
                graph: &self.graph,
                seeds: &self.graph.entry_points,
                ef,
                budget,
                top_k,
                n: self.n,
            },
            Some(&stage05),
            if use_radial_bound {
                Some(&prune_lb)
            } else {
                None
            },
            &prefetch,
            &beam_dist,
            &exact_dist,
        )
    }
}

/// Tier-2 基线包装器：支持 (1) 欧氏 L2 构图/检索 + 精确重排，(2) 全局单常曲率 Poincaré SM + 精确重排，(3) 全精度精确乘积 Vamana。
pub struct ProductBaselineIndex {
    pub factors: Vec<ProductFactor>,
    pub total_dim: usize,
    pub vectors: Vec<f32>,
    pub n: usize,
    pub graph: VarGraph,
    mode: ProductBaselineMode,
}

#[derive(Clone, Copy)]
pub enum ProductBaselineMode {
    EuclideanL2,
    ExactProduct,
}

struct SingleBallState {
    store: Bq2Store,
    metas: Vec<(f32, f32, SmCodeParams)>,
    curvature: f64,
    scale: f64,
}

pub struct ProductSingleBallIndex {
    pub factors: Vec<ProductFactor>,
    pub total_dim: usize,
    pub vectors: Vec<f32>,
    pub n: usize,
    state: SingleBallState,
    pub graph: VarGraph,
}

impl ProductSingleBallIndex {
    pub fn build(
        factors: Vec<ProductFactor>,
        vectors: &[f32],
        curvature: f64,
        m_max: usize,
        ef_construction: usize,
        alpha: f64,
    ) -> Self {
        let total_dim: usize = factors.iter().map(|f| f.dim()).sum();
        let n = vectors.len() / total_dim;
        // 计算安全缩放因子，使全向量映射进单一曲率 -curvature 的 Poincaré 球内 (sqrt(c)*r <= 0.96)
        let max_norm = vectors
            .chunks_exact(total_dim)
            .map(|r| norm_squared_f64(r).sqrt())
            .fold(0.0f64, f64::max)
            .max(1e-6);
        let max_ball_r = 0.96 / curvature.sqrt();
        let scale = max_ball_r / max_norm;

        let scaled: Vec<f32> = vectors
            .iter()
            .map(|&x| (f64::from(x) * scale) as f32)
            .collect();
        let store = Bq2Store::from_vectors_parallel(&scaled, total_dim);
        let metas: Vec<(f32, f32, SmCodeParams)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &scaled[i * total_dim..(i + 1) * total_dim];
                let (sm, r) = SmCodeParams::from_vector_and_store(row, &store, i);
                let f = (1.0 - curvature * r * r).max(1e-12).recip() as f32;
                (r as f32, f, sm)
            })
            .collect();

        let state = SingleBallState {
            store,
            metas,
            curvature,
            scale,
        };
        let dist_fn = |a: usize, b: usize| -> f64 {
            let (ra, fa, sma) = state.metas[a];
            let (rb, fb, smb) = state.metas[b];
            let counts = state.store.sm_counts(a, b, total_dim);
            let cos_hat = sm_calibrated_cosine_fast(sma, smb, counts, total_dim);
            let dr = f64::from(ra) - f64::from(rb);
            (dr * dr + 2.0 * f64::from(ra) * f64::from(rb) * (1.0 - cos_hat)).max(0.0)
                * f64::from(fa)
                * f64::from(fb)
        };
        let radii: Vec<f64> = scaled
            .chunks_exact(total_dim)
            .map(|r| norm_squared_f64(r).sqrt())
            .collect();
        let entries = pick_diverse_hubs(&scaled, total_dim, &radii, 8);
        let graph = VarGraph::build(n, m_max, ef_construction, alpha, entries, dist_fn);
        Self {
            factors,
            total_dim,
            vectors: vectors.to_vec(),
            n,
            state,
            graph,
        }
    }

    pub fn search(&self, query: &[f32], top_k: usize, ef: usize, budget: usize) -> SearchOutcome {
        let q_scaled: Vec<f32> = query
            .iter()
            .map(|&x| (f64::from(x) * self.state.scale) as f32)
            .collect();
        let q_sig = Bq2Signature::from_vector(&q_scaled);
        let (q_sm, q_r) = SmCodeParams::from_vector_and_sig(&q_scaled, &q_sig);
        let q_f = (1.0 - self.state.curvature * q_r * q_r).max(1e-12).recip();
        let mut q_asym = AsymQuerySm::new();
        q_asym.encode_into(&q_scaled, q_r);

        let beam_dist = |node: usize| -> f64 {
            let (vr, vf, vsm) = self.state.metas[node];
            let counts = self
                .state
                .store
                .sm_counts_to_sig(node, &q_sig, self.total_dim);
            let cos_hat = sm_calibrated_cosine_fast(q_sm, vsm, counts, self.total_dim);
            let dr = q_r - f64::from(vr);
            (dr * dr + 2.0 * q_r * f64::from(vr) * (1.0 - cos_hat)).max(0.0) * q_f * f64::from(vf)
        };
        let stage05 = |node: usize| -> f64 {
            let (vr, vf, vsm) = self.state.metas[node];
            let cos_hat = q_asym.calibrated_cosine(&self.state.store, node, vsm);
            let dr = q_r - f64::from(vr);
            (dr * dr + 2.0 * q_r * f64::from(vr) * (1.0 - cos_hat)).max(0.0) * q_f * f64::from(vf)
        };
        let exact_dist = |node: usize| -> f64 {
            let row = &self.vectors[node * self.total_dim..(node + 1) * self.total_dim];
            exact_product_distance_sq(&self.factors, query, row)
        };
        search_graph_with_rerank(
            SearchRequest {
                graph: &self.graph,
                seeds: &self.graph.entry_points,
                ef,
                budget,
                top_k,
                n: self.n,
            },
            Some(&stage05),
            None::<&fn(usize, f64) -> bool>,
            &nop_prefetch,
            &beam_dist,
            &exact_dist,
        )
    }

    /// 返回单球基线热索引（量化签名 + 元数据 + 邻接表）占用字节数。
    pub fn index_memory_bytes(&self) -> usize {
        let sig = self.state.store.len() * self.state.store.chunks() * 16;
        let meta = self.state.metas.len() * std::mem::size_of::<(f32, f32, SmCodeParams)>();
        let graph: usize = self.graph.neighbors.iter().map(|v| v.len() * 4).sum();
        sig + meta + graph
    }
}

impl ProductBaselineIndex {
    pub fn build(
        factors: Vec<ProductFactor>,
        vectors: &[f32],
        mode: ProductBaselineMode,
        m_max: usize,
        ef_construction: usize,
        alpha: f64,
    ) -> Self {
        let total_dim: usize = factors.iter().map(|f| f.dim()).sum();
        let n = vectors.len() / total_dim;
        let dist_fn = |a: usize, b: usize| -> f64 {
            let va = &vectors[a * total_dim..(a + 1) * total_dim];
            let vb = &vectors[b * total_dim..(b + 1) * total_dim];
            match mode {
                ProductBaselineMode::EuclideanL2 => poincare_proxy(va, vb, 1.0),
                ProductBaselineMode::ExactProduct => exact_product_distance_sq(&factors, va, vb),
            }
        };
        let scores: Vec<f64> = vectors
            .chunks_exact(total_dim)
            .map(|row| match mode {
                ProductBaselineMode::EuclideanL2 => norm_squared_f64(row).sqrt(),
                ProductBaselineMode::ExactProduct => {
                    let mut off = 0usize;
                    let mut depth = 0.0f64;
                    for &f in &factors {
                        let d = f.dim();
                        if let ProductFactor::Hyperbolic { curvature, .. } = f {
                            let s = curvature.sqrt();
                            let r = norm_squared_f64(&row[off..off + d]).sqrt();
                            depth += (2.0 * (s * r).min(0.999_999).atanh()) / s;
                        }
                        off += d;
                    }
                    depth
                }
            })
            .collect();
        let entries = pick_diverse_hubs(vectors, total_dim, &scores, 8);
        let graph = VarGraph::build(n, m_max, ef_construction, alpha, entries, dist_fn);
        Self {
            factors,
            total_dim,
            vectors: vectors.to_vec(),
            n,
            graph,
            mode,
        }
    }

    pub fn search(&self, query: &[f32], top_k: usize, ef: usize, budget: usize) -> SearchOutcome {
        let exact_dist = |node: usize| -> f64 {
            let row = &self.vectors[node * self.total_dim..(node + 1) * self.total_dim];
            exact_product_distance_sq(&self.factors, query, row)
        };
        let beam_dist = |node: usize| -> f64 {
            let row = &self.vectors[node * self.total_dim..(node + 1) * self.total_dim];
            match self.mode {
                ProductBaselineMode::EuclideanL2 => poincare_proxy(query, row, 1.0),
                ProductBaselineMode::ExactProduct => {
                    exact_product_distance_sq(&self.factors, query, row)
                }
            }
        };
        let eff_budget = match self.mode {
            ProductBaselineMode::ExactProduct => ef.max(budget),
            ProductBaselineMode::EuclideanL2 => budget,
        };
        search_graph_with_rerank(
            SearchRequest {
                graph: &self.graph,
                seeds: &self.graph.entry_points,
                ef,
                budget: eff_budget,
                top_k,
                n: self.n,
            },
            None::<&fn(usize) -> f64>,
            None::<&fn(usize, f64) -> bool>,
            &nop_prefetch,
            &beam_dist,
            &exact_dist,
        )
    }

    /// 返回基线索引热路径（不含外部重排原向量时的图结构）或包含原向量时的内存占用。
    pub fn index_memory_bytes(&self) -> usize {
        self.graph.neighbors.iter().map(|v| v.len() * 4).sum()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  层级 3：径向变曲率翘曲乘积流形 ds^2 = dR^2 + φ(R)^2 dΩ^2
// ═════════════════════════════════════════════════════════════════════════════

/// 球对称翘曲乘积流形径向缩放剖面 φ(R)。
///
/// 满足 φ(0) = 0, φ'(0) = 1, φ(R) > 0 且严格凸（φ''(R) > 0 => K_rad(R) = -φ''(R)/φ(R) < 0）。
#[derive(Clone, Copy, Debug)]
pub enum WarpedProfile {
    /// 常负曲率特例 φ(R) = sinh(sqrt(c)*R)/sqrt(c)，专用于验证 Clairaut 数值求积与闭式解一致。
    ConstantHyperbolic { curvature: f64 },
    /// 外层逐渐变平（核心强双曲、外围渐平）：φ(R) = R + α R^3，K_rad(R) = -6α / (1 + α R^2)
    Softening { alpha: f64 },
    /// 外层曲率逐渐变陡（双曲喇叭型）：φ(R) = R exp(α R^2)，K_rad(R) = -(6α + 4α^2 R^2)
    Steepening { alpha: f64 },
}

impl WarpedProfile {
    #[inline]
    pub fn phi(self, r: f64) -> f64 {
        match self {
            Self::ConstantHyperbolic { curvature } => {
                let s = curvature.sqrt();
                (s * r).sinh() / s
            }
            Self::Softening { alpha } => r + alpha * r * r * r,
            Self::Steepening { alpha } => r * (alpha * r * r).exp(),
        }
    }

    #[inline]
    pub fn radial_curvature(self, r: f64) -> f64 {
        match self {
            Self::ConstantHyperbolic { curvature } => -curvature,
            Self::Softening { alpha } => -6.0 * alpha / (1.0 + alpha * r * r),
            Self::Steepening { alpha } => -(6.0 * alpha + 4.0 * alpha * alpha * r * r),
        }
    }
}

/// 16 点 Gauss-Legendre 求积节点与权重（在 [0, 1] 上的对称半区间缩放至 [-1, 1]）。
const GL_X: [f64; 8] = [
    0.09501250983763744,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.755404408355003,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
];
const GL_W: [f64; 8] = [
    0.1894506104550685,
    0.18260341504492358,
    0.16915651939500254,
    0.14959598881657674,
    0.12462897125553388,
    0.09515851168249279,
    0.062253523938647894,
    0.027152459411754096,
];

const PANELS: [(f64, f64); 4] = [(0.0, 0.04), (0.04, 0.16), (0.16, 0.45), (0.45, 1.0)];

const fn build_clairaut_quad_table() -> ([f64; 64], [f64; 64]) {
    let mut u2 = [0.0f64; 64];
    let mut wu2 = [0.0f64; 64];
    let mut pi = 0usize;
    while pi < 4 {
        let (u0, u1) = PANELS[pi];
        let mid = 0.5 * (u0 + u1);
        let half = 0.5 * (u1 - u0);
        let mut k = 0usize;
        while k < 8 {
            let w = GL_W[k] * half;
            let dx = half * GL_X[k];
            let ua = mid - dx;
            let ub = mid + dx;
            let idx0 = pi * 16 + k * 2;
            u2[idx0] = ua * ua;
            wu2[idx0] = 2.0 * w * ua;
            u2[idx0 + 1] = ub * ub;
            wu2[idx0 + 1] = 2.0 * w * ub;
            k += 1;
        }
        pi += 1;
    }
    (u2, wu2)
}

const CLAIRAUT_QUAD: ([f64; 64], [f64; 64]) = build_clairaut_quad_table();

#[inline(always)]
fn clairaut_quad<F>(profile: WarpedProfile, a: f64, b: f64, ang_mom: f64, integrand: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let span = b - a;
    if span <= 0.0 {
        return 0.0;
    }
    let l2 = ang_mom * ang_mom;
    let (u2_tbl, wu2_tbl) = &CLAIRAUT_QUAD;
    let mut sum = 0.0f64;
    for i in 0..64 {
        let r = a + span * u2_tbl[i];
        let p = profile.phi(r);
        let root = (p * p - l2).max(1e-30).sqrt();
        sum += wu2_tbl[i] * integrand(p, root);
    }
    sum * span
}

/// 仅计算角度积分项 Δθ（供 Brent 求根迭代内部调用，省去 30% 弧长浮点开销）。
#[inline]
fn clairaut_theta(profile: WarpedProfile, a: f64, b: f64, ang_mom: f64) -> f64 {
    clairaut_quad(profile, a, b, ang_mom, |p, root| 1.0 / (p * root)) * ang_mom
}

/// 仅计算测地弧长积分项（仅在根收敛后计算一次）。
#[inline]
fn clairaut_length(profile: WarpedProfile, a: f64, b: f64, ang_mom: f64) -> f64 {
    clairaut_quad(profile, a, b, ang_mom, |p, root| p / root)
}

/// 分段标量 Brent / 保护二分求根器：求解单调函数 f(x) = target 在 [lo, hi] 上的根。
fn solve_monotone<F>(lo: f64, hi: f64, f_lo: f64, f_hi: f64, target: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut a = lo;
    let mut b = hi;
    let mut fa = f_lo - target;
    let mut fb = f_hi - target;
    if fa.abs() < 1e-12 {
        return a;
    }
    if fb.abs() < 1e-12 {
        return b;
    }
    if fa.signum() == fb.signum() {
        return if fa.abs() <= fb.abs() { a } else { b };
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    for _ in 0..48 {
        if fb == 0.0 {
            break;
        }
        if fc.signum() == fb.signum() {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        // 由变分费马原理 ∂L/∂r_* = 0，角动量/折返半径误差 2e-12 在测地弧长上保证 < 1e-11 严格一致性
        let tol = 2.0 * 1e-12 * b.abs().max(1.0);
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            break;
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q) = if (a - c).abs() < 1e-15 {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q0 = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q0 * (q0 - r) - (b - a) * (r - 1.0)),
                    (q0 - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }
        a = b;
        fa = fb;
        b += if d.abs() > tol { d } else { tol.copysign(m) };
        fb = f(b) - target;
    }
    b
}

/// 利用 Clairaut 守恒积分计算径向变曲率翘曲乘积流形的精确黎曼测地距离。
pub fn clairaut_distance(profile: WarpedProfile, rq: f64, rv: f64, cos_theta: f64) -> f64 {
    if rq <= 1e-12 {
        return rv.max(0.0);
    }
    if rv <= 1e-12 {
        return rq.max(0.0);
    }
    let cos_c = cos_theta.clamp(-1.0, 1.0);
    if cos_c >= 1.0 - 1e-14 {
        return (rq - rv).abs();
    }
    if cos_c <= -1.0 + 1e-14 {
        return rq + rv;
    }
    let theta = cos_c.acos();
    let (r_lo, r_hi) = if rq <= rv { (rq, rv) } else { (rv, rq) };
    let l_max = profile.phi(r_lo);

    // 单调分支最大角度（当 r_lo < r_hi 时从 r_lo 出发直达 r_hi）
    let theta_crit = if r_hi > r_lo + 1e-13 {
        clairaut_theta(profile, r_lo, r_hi, l_max)
    } else {
        0.0
    };

    if theta <= theta_crit && r_hi > r_lo + 1e-13 {
        // Case 1: 径向单调上升段 L ∈ [0, phi(r_lo)]
        let eval_theta = |ang_mom: f64| clairaut_theta(profile, r_lo, r_hi, ang_mom);
        let l_star = solve_monotone(0.0, l_max, 0.0, theta_crit, theta, eval_theta);
        return clairaut_length(profile, r_lo, r_hi, l_star);
    }

    // Case 2: 测地线先向内折返到近日点 r_* ∈ [0, r_lo]，再向外爬升经过 r_lo 至 r_hi
    if (theta - theta_crit).abs() <= 1e-14 {
        return clairaut_length(profile, r_lo, r_hi, l_max);
    }
    let eval_dip_theta = |r_turn: f64| -> f64 {
        if r_turn <= 1e-13 {
            return std::f64::consts::PI;
        }
        let l = profile.phi(r_turn);
        let t_inner = clairaut_theta(profile, r_turn, r_lo, l);
        let t_outer = if r_hi > r_lo + 1e-13 {
            clairaut_theta(profile, r_lo, r_hi, l)
        } else {
            0.0
        };
        2.0 * t_inner + t_outer
    };
    let r_star = solve_monotone(
        0.0,
        r_lo,
        std::f64::consts::PI,
        theta_crit,
        theta,
        eval_dip_theta,
    );
    let l = profile.phi(r_star);
    let len_inner = clairaut_length(profile, r_star, r_lo, l);
    let len_outer = if r_hi > r_lo + 1e-13 {
        clairaut_length(profile, r_lo, r_hi, l)
    } else {
        0.0
    };
    2.0 * len_inner + len_outer
}

/// 离线为特定 φ(R) 预校准的二维/三维径向-半正矢角表结构（驻留 L2/L1，彻底消除束搜索内的积分）。
///
/// 在横轴 `u = (1 - cos_theta) * 0.5 = sin^2(theta/2)` 上采样（近角区呈线性行为且热路径零开方）。
pub struct WarpedProxyTable {
    nr: usize,
    ns: usize,
    r_max: f64,
    inv_dr: f64,
    data: Vec<f32>,
}

/// 单条查询按给定 `R_q` 切出的 1D-径向 × 半正矢角 L1 常驻切片（~16 KB）。
struct QueryWarpedSlice {
    nr: usize,
    ns: usize,
    inv_dr: f32,
    r_max: f32,
    q_r: f32,
    slice: Vec<f32>,
}

impl WarpedProxyTable {
    pub fn build(profile: WarpedProfile, r_max: f64, nr: usize, ns: usize) -> Self {
        let r_max = r_max.max(0.1);
        let mut pairs = Vec::with_capacity(nr * (nr + 1) / 2);
        for i in 0..nr {
            for j in i..nr {
                pairs.push((i, j));
            }
        }
        let step_r = r_max / ((nr - 1) as f64);
        let step_u = 1.0 / ((ns - 1) as f64);
        let computed: Vec<(usize, usize, Vec<f32>)> = pairs
            .into_par_iter()
            .map(|(i, j)| {
                let ri = i as f64 * step_r;
                let rj = j as f64 * step_r;
                let radial_sq = (ri - rj) * (ri - rj);
                let mut row = Vec::with_capacity(ns);
                for k in 0..ns {
                    let u = k as f64 * step_u;
                    let cos_t = (1.0 - 2.0 * u).clamp(-1.0, 1.0);
                    let d = clairaut_distance(profile, ri, rj, cos_t);
                    row.push((d * d - radial_sq).max(0.0) as f32);
                }
                (i, j, row)
            })
            .collect();

        let mut data = vec![0.0f32; nr * nr * ns];
        for (i, j, row) in computed {
            let off_ij = (i * nr + j) * ns;
            let off_ji = (j * nr + i) * ns;
            data[off_ij..off_ij + ns].copy_from_slice(&row);
            data[off_ji..off_ji + ns].copy_from_slice(&row);
        }
        Self {
            nr,
            ns,
            r_max,
            inv_dr: ((nr - 1) as f64) / r_max,
            data,
        }
    }

    #[inline]
    fn eval(&self, rq: f64, rv: f64, cos_hat: f64) -> f64 {
        let u = ((1.0 - cos_hat) * 0.5).clamp(0.0, 1.0);
        let qi = (rq * self.inv_dr).clamp(0.0, (self.nr - 1) as f64);
        let vi = (rv * self.inv_dr).clamp(0.0, (self.nr - 1) as f64);
        let si = (u * ((self.ns - 1) as f64)).clamp(0.0, (self.ns - 1) as f64);
        let i0 = (qi.floor() as usize).min(self.nr - 2);
        let j0 = (vi.floor() as usize).min(self.nr - 2);
        let k0 = (si.floor() as usize).min(self.ns - 2);
        let fq = qi - i0 as f64;
        let fv = vi - j0 as f64;
        let fs = si - k0 as f64;

        let get = |ii: usize, jj: usize, kk: usize| -> f64 {
            f64::from(self.data[(ii * self.nr + jj) * self.ns + kk])
        };
        let interp_s = |ii: usize, jj: usize| -> f64 {
            let c0 = get(ii, jj, k0);
            let c1 = get(ii, jj, k0 + 1);
            c0 + fs * (c1 - c0)
        };
        let a00 = interp_s(i0, j0);
        let a01 = interp_s(i0, j0 + 1);
        let a10 = interp_s(i0 + 1, j0);
        let a11 = interp_s(i0 + 1, j0 + 1);
        let a0 = a00 + fv * (a01 - a00);
        let a1 = a10 + fv * (a11 - a10);
        let ang_excess = (a0 + fq * (a1 - a0)).max(0.0);
        let dr = rq - rv;
        dr * dr + ang_excess
    }

    /// 每次查询开始前沿 `R_q` 切出一段仅 `nr * ns` 项（~16 KB）的紧凑 L1 切片。
    fn slice_for_query(&self, rq: f64) -> QueryWarpedSlice {
        let qi = (rq * self.inv_dr).clamp(0.0, (self.nr - 1) as f64);
        let i0 = (qi.floor() as usize).min(self.nr - 2);
        let fq = (qi - i0 as f64) as f32;
        let row0 = &self.data[i0 * self.nr * self.ns..(i0 + 1) * self.nr * self.ns];
        let row1 = &self.data[(i0 + 1) * self.nr * self.ns..(i0 + 2) * self.nr * self.ns];
        let mut slice = Vec::with_capacity(self.nr * self.ns);
        for (&a, &b) in row0.iter().zip(row1.iter()) {
            slice.push(a + fq * (b - a));
        }
        QueryWarpedSlice {
            nr: self.nr,
            ns: self.ns,
            inv_dr: self.inv_dr as f32,
            r_max: self.r_max as f32,
            q_r: rq as f32,
            slice,
        }
    }
}

impl QueryWarpedSlice {
    #[inline]
    fn eval(&self, rv: f32, cos_hat: f64) -> f64 {
        let u = (((1.0 - cos_hat) * 0.5).clamp(0.0, 1.0)) as f32;
        let vi = (rv.clamp(0.0, self.r_max) * self.inv_dr).min((self.nr - 1) as f32);
        let si = (u * ((self.ns - 1) as f32)).min((self.ns - 1) as f32);
        let j0 = (vi.floor() as usize).min(self.nr - 2);
        let k0 = (si.floor() as usize).min(self.ns - 2);
        let fv = vi - j0 as f32;
        let fs = si - k0 as f32;

        let b0 = j0 * self.ns + k0;
        let b1 = (j0 + 1) * self.ns + k0;
        // SAFETY：j0 <= nr-2, k0 <= ns-2 均严格保证在 slice 边界内
        let (v00, v01, v10, v11) = unsafe {
            (
                *self.slice.get_unchecked(b0),
                *self.slice.get_unchecked(b0 + 1),
                *self.slice.get_unchecked(b1),
                *self.slice.get_unchecked(b1 + 1),
            )
        };
        let a0 = v00 + fs * (v01 - v00);
        let a1 = v10 + fs * (v11 - v10);
        let ang_excess = f64::from((a0 + fv * (a1 - a0)).max(0.0));
        let dr = f64::from(self.q_r - rv);
        dr * dr + ang_excess
    }
}

/// 层级 3 径向变曲率翘曲流形对照方案。
#[derive(Clone, Copy, Debug)]
pub enum Tier3Method {
    /// 欧氏向量 L2 导航 + Clairaut 精确重排
    EuclideanL2,
    /// 强行固定单常曲率 Poincaré SM-2bit + Clairaut 精确重排
    FixedConstantHyperbolicSm { curvature: f64 },
    /// 无离线查表的局部共形/中点近似 d^2 ≈ ΔR^2 + 2 φ((R_q+R_v)/2)^2 (1 - cos_hat) + SM-2bit
    LocalWarpedFormulaSm,
    /// Clairaut 预校准 2D 查询切片查表 + 极坐标/残差锥 SM-2bit + Stage-0.5 + 几何剪枝（Cauchy-Schwarz 球面角下界 + 径向/Dipping-Horizon 下界）+ Clairaut 精确重排
    WarpedTableSm,
    /// 全精度 Clairaut 构图与导航（理论精排上界对照）
    ExactClairaut,
}

/// 紧凑 16B 热路径束搜索节点头（4 个节点恰好占满单条 64B CPU 缓存行）。
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct WarpedBeamMeta {
    r: f32,
    cone_alpha: f32,
    cone_beta: f32,
    cone_id: u32,
}

/// 紧凑 24B 极坐标节点元数据与共形标量（分离 16B 热束头部以提升 L1/L2 缓存驻留）。
#[repr(C)]
#[derive(Clone, Copy)]
struct WarpedNodeMeta {
    sm_weak: f32,
    sm_delta: f32,
    sm_sigma: f32,
    strong_ones: u32,
    r_ball: f32,
    bf_fixed: f32,
}

impl WarpedNodeMeta {
    #[inline(always)]
    fn sm_params(self) -> SmCodeParams {
        SmCodeParams {
            weak: self.sm_weak,
            delta: self.sm_delta,
            sigma: self.sm_sigma,
            strong_ones: self.strong_ones,
        }
    }
}

/// 层级 3 50k 实验统一索引，按外在笛卡尔参数存储坐标 `x = R * u`（`R = ||x||` 恰好就是流形测地半径 `R`！）。
///
/// 这也契合流形极坐标测地法坐标系（Normal/Polar Geodesic Coordinates）：原点到点的测地半径 `R` 即 `||x||_2`，方向 `u = x / ||x||_2`。
pub struct WarpedManifoldIndex {
    pub profile: WarpedProfile,
    pub method: Tier3Method,
    pub dim: usize,
    pub n: usize,
    pub vectors: Vec<f32>,
    store: Option<Bq2Store>,
    beam_metas: Vec<WarpedBeamMeta>,
    sm_metas: Vec<WarpedNodeMeta>,
    cones: Option<WarpedConeDict>,
    table: Option<WarpedProxyTable>,
    r_max: f64,
    pub graph: VarGraph,
}

#[derive(Clone, Copy, Debug)]
pub struct GraphBuildConfig {
    pub m_max: usize,
    pub ef_construction: usize,
    pub alpha: f64,
    pub use_compass: bool,
    pub use_cones: bool,
}

impl WarpedManifoldIndex {
    #[inline]
    pub fn exact_distance(&self, a: &[f32], b: &[f32]) -> f64 {
        let ra = norm_squared_f64(a).sqrt();
        let rb = norm_squared_f64(b).sqrt();
        let cos_t = if ra > 1e-12 && rb > 1e-12 {
            (dot_f32_f64(a, b) / (ra * rb)).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        clairaut_distance(self.profile, ra, rb, cos_t)
    }

    pub fn build(
        profile: WarpedProfile,
        method: Tier3Method,
        dim: usize,
        vectors: &[f32],
        m_max: usize,
        ef_construction: usize,
        alpha: f64,
    ) -> Self {
        Self::build_with_config(
            profile,
            method,
            dim,
            vectors,
            GraphBuildConfig {
                m_max,
                ef_construction,
                alpha,
                use_compass: false,
                use_cones: true,
            },
        )
    }

    pub fn build_with_config(
        profile: WarpedProfile,
        method: Tier3Method,
        dim: usize,
        vectors: &[f32],
        cfg: GraphBuildConfig,
    ) -> Self {
        let GraphBuildConfig {
            m_max,
            ef_construction,
            alpha,
            use_compass,
            use_cones,
        } = cfg;
        assert_eq!(vectors.len() % dim, 0);
        let n = vectors.len() / dim;
        let radii: Vec<f64> = vectors
            .par_chunks_exact(dim)
            .map(|r| norm_squared_f64(r).sqrt())
            .collect();
        let r_max = radii.iter().copied().fold(0.0f64, f64::max) * 1.02 + 1e-3;

        let fixed_curvature = match method {
            Tier3Method::FixedConstantHyperbolicSm { curvature } => curvature,
            _ => 1.0,
        };
        let fixed_root = fixed_curvature.sqrt();

        let table = if matches!(method, Tier3Method::WarpedTableSm) {
            Some(WarpedProxyTable::build(profile, r_max, 64, 65))
        } else {
            None
        };
        // 对 50k 规模 ExactClairaut 图构建先建一张高密度表供 Vamana 连边查表加速，查询期再走精确 Clairaut
        let oracle_build_table = if matches!(method, Tier3Method::ExactClairaut) && n > 2_000 {
            Some(WarpedProxyTable::build(profile, r_max, 128, 65))
        } else {
            None
        };

        let needs_sm = matches!(
            method,
            Tier3Method::FixedConstantHyperbolicSm { .. }
                | Tier3Method::LocalWarpedFormulaSm
                | Tier3Method::WarpedTableSm
        );
        let cone_fit = if needs_sm && use_cones {
            WarpedConeDict::fit_and_encode(vectors, dim, n)
        } else {
            None
        };
        let (cones, cone_meta, encode_vecs) = if let Some(fit) = cone_fit {
            (Some(fit.dict), fit.metas, Some(fit.res_vecs))
        } else {
            (None, Vec::new(), None)
        };

        let store = if needs_sm {
            let src = encode_vecs.as_deref().unwrap_or(vectors);
            Some(Bq2Store::from_vectors_parallel(src, dim))
        } else {
            None
        };

        let (beam_metas, sm_metas): (Vec<WarpedBeamMeta>, Vec<WarpedNodeMeta>) =
            if let Some(ref st) = store {
                let src = encode_vecs.as_deref().unwrap_or(vectors);
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let row = &src[i * dim..(i + 1) * dim];
                        let r = radii[i];
                        let (sm, _) = SmCodeParams::from_vector_and_store(row, st, i);
                        let r_ball = (0.5 * fixed_root * r).tanh() / fixed_root;
                        let bf_fixed =
                            (1.0 - fixed_curvature * r_ball * r_ball).max(1e-12).recip() as f32;
                        let (cone_id, cone_alpha, cone_beta) = if cones.is_some() {
                            cone_meta[i]
                        } else {
                            (u32::MAX, 0.0, 1.0)
                        };
                        (
                            WarpedBeamMeta {
                                r: r as f32,
                                cone_alpha,
                                cone_beta,
                                cone_id,
                            },
                            WarpedNodeMeta {
                                sm_weak: sm.weak,
                                sm_delta: sm.delta,
                                sm_sigma: sm.sigma,
                                strong_ones: sm.strong_ones,
                                r_ball: r_ball as f32,
                                bf_fixed,
                            },
                        )
                    })
                    .unzip()
            } else {
                let bms = radii
                    .iter()
                    .map(|&r| WarpedBeamMeta {
                        r: r as f32,
                        cone_alpha: 0.0,
                        cone_beta: 1.0,
                        cone_id: u32::MAX,
                    })
                    .collect();
                (bms, Vec::new())
            };

        // 公平选取靠近中内层且方向分散的 8 个核心入口枢纽
        let entries = pick_diverse_hubs(vectors, dim, &radii, 8);

        let pair_sm_cos = |a: usize, b: usize| -> f64 {
            let st = store.as_ref().unwrap();
            let bma = beam_metas[a];
            let bmb = beam_metas[b];
            if let Some(ref dict) = cones {
                if bma.cone_id == bmb.cone_id {
                    let cos_res = if bma.cone_beta <= 1e-6 || bmb.cone_beta <= 1e-6 {
                        0.0
                    } else {
                        let counts = st.sm_counts(a, b, dim);
                        sm_calibrated_cosine_fast(
                            sm_metas[a].sm_params(),
                            sm_metas[b].sm_params(),
                            counts,
                            dim,
                        )
                    };
                    (f64::from(bma.cone_alpha) * f64::from(bmb.cone_alpha)
                        + f64::from(bma.cone_beta) * f64::from(bmb.cone_beta) * cos_res)
                        .clamp(-1.0, 1.0)
                } else {
                    (f64::from(bma.cone_alpha)
                        * f64::from(bmb.cone_alpha)
                        * dict.axis_dot(bma.cone_id, bmb.cone_id))
                    .clamp(-1.0, 1.0)
                }
            } else {
                let counts = st.sm_counts(a, b, dim);
                sm_calibrated_cosine_fast(
                    sm_metas[a].sm_params(),
                    sm_metas[b].sm_params(),
                    counts,
                    dim,
                )
            }
        };

        let pair_dist = |a: usize, b: usize| -> f64 {
            match method {
                Tier3Method::EuclideanL2 => {
                    let va = &vectors[a * dim..(a + 1) * dim];
                    let vb = &vectors[b * dim..(b + 1) * dim];
                    poincare_proxy(va, vb, 1.0)
                }
                Tier3Method::ExactClairaut => {
                    let va = &vectors[a * dim..(a + 1) * dim];
                    let vb = &vectors[b * dim..(b + 1) * dim];
                    let ra = radii[a];
                    let rb = radii[b];
                    let cos_t = if ra > 1e-12 && rb > 1e-12 {
                        (dot_f32_f64(va, vb) / (ra * rb)).clamp(-1.0, 1.0)
                    } else {
                        1.0
                    };
                    if let Some(ref tbl) = oracle_build_table {
                        tbl.eval(ra, rb, cos_t)
                    } else {
                        let d = clairaut_distance(profile, ra, rb, cos_t);
                        d * d
                    }
                }
                Tier3Method::FixedConstantHyperbolicSm { .. } => {
                    let ma = sm_metas[a];
                    let mb = sm_metas[b];
                    let cos_hat = pair_sm_cos(a, b);
                    let ra = f64::from(ma.r_ball);
                    let rb = f64::from(mb.r_ball);
                    let dr = ra - rb;
                    (dr * dr + 2.0 * ra * rb * (1.0 - cos_hat)).max(0.0)
                        * f64::from(ma.bf_fixed)
                        * f64::from(mb.bf_fixed)
                }
                Tier3Method::LocalWarpedFormulaSm => {
                    let ra = f64::from(beam_metas[a].r);
                    let rb = f64::from(beam_metas[b].r);
                    let cos_hat = pair_sm_cos(a, b);
                    let dr = ra - rb;
                    let phi_mid = profile.phi(0.5 * (ra + rb));
                    dr * dr + 2.0 * phi_mid * phi_mid * (1.0 - cos_hat).max(0.0)
                }
                Tier3Method::WarpedTableSm => {
                    let tbl = table.as_ref().unwrap();
                    let cos_hat = pair_sm_cos(a, b);
                    tbl.eval(
                        f64::from(beam_metas[a].r),
                        f64::from(beam_metas[b].r),
                        cos_hat,
                    )
                }
            }
        };

        let radial_dist_fn = |a: usize, b: usize| -> f64 { (radii[a] - radii[b]).abs() };
        let angular_loss_fn = |a: usize, b: usize| -> f64 {
            if store.is_some() {
                (1.0 - pair_sm_cos(a, b)).clamp(0.0, 2.0)
            } else {
                let va = &vectors[a * dim..(a + 1) * dim];
                let vb = &vectors[b * dim..(b + 1) * dim];
                let r_prod = (radii[a] * radii[b]).max(1e-12);
                (1.0 - (dot_f32_f64(va, vb) / r_prod).clamp(-1.0, 1.0)).clamp(0.0, 2.0)
            }
        };
        let quadri_spec = QuadriPruneSpec {
            depths: &radii,
            radial_dist_fn: &radial_dist_fn,
            angular_loss_fn: &angular_loss_fn,
            iso_tol: 0.60,
            elev_min: 0.18,
            elev_step: 0.65,
            geo_fraction: 0.375,
            ang_guard_delta: 0.45,
            elev_ang_delta: 0.50,
            iso_score_uses_geodesic: false,
            bridge_cap: 0,
        };

        // 当流形不存在可对齐的球面残差锥（各向同性连续分布）时启用 QuadriPrune 等深环破解原点黑洞遮蔽；残差锥簇下沿用纯净 α-RNG
        let use_quadri =
            use_compass && !matches!(method, Tier3Method::EuclideanL2) && cones.is_none();
        let graph = VarGraph::build_with_quadri(
            n,
            m_max,
            ef_construction,
            alpha,
            entries,
            pair_dist,
            if use_quadri { Some(&quadri_spec) } else { None },
        );

        Self {
            profile,
            method,
            dim,
            n,
            vectors: vectors.to_vec(),
            store,
            beam_metas,
            sm_metas,
            cones,
            table,
            r_max,
            graph,
        }
    }

    pub fn index_memory_bytes(&self) -> usize {
        let sig = self.store.as_ref().map_or(0, |s| s.len() * s.chunks() * 16);
        let bm = self.beam_metas.len() * std::mem::size_of::<WarpedBeamMeta>();
        let sm = self.sm_metas.len() * std::mem::size_of::<WarpedNodeMeta>();
        let cone_b = self.cones.as_ref().map_or(0, |c| c.memory_bytes());
        let tbl = self.table.as_ref().map_or(0, |t| t.data.len() * 4);
        let graph_b: usize = self.graph.neighbors.iter().map(|v| v.len() * 4).sum();
        sig + bm + sm + cone_b + tbl + graph_b
    }

    #[inline(always)]
    fn prefetch_beam_node(&self, node: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let p = self.beam_metas.as_ptr().add(node) as *const i8;
            _mm_prefetch(p, _MM_HINT_T0);
            if self.cones.is_none()
                && let Some(ref st) = self.store
            {
                st.prefetch_sig(node);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = node;
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef: usize,
        budget: usize,
        use_radial_bound: bool,
    ) -> SearchOutcome {
        let rq = norm_squared_f64(query).sqrt();
        let exact_dist = |node: usize| -> f64 {
            let row = &self.vectors[node * self.dim..(node + 1) * self.dim];
            self.exact_distance(query, row)
        };

        if matches!(
            self.method,
            Tier3Method::EuclideanL2 | Tier3Method::ExactClairaut
        ) {
            let beam = |node: usize| -> f64 {
                let row = &self.vectors[node * self.dim..(node + 1) * self.dim];
                match self.method {
                    Tier3Method::EuclideanL2 => poincare_proxy(query, row, 1.0),
                    Tier3Method::ExactClairaut => {
                        let rb = f64::from(self.beam_metas[node].r);
                        let cos_t = if rq > 1e-12 && rb > 1e-12 {
                            (dot_f32_f64(query, row) / (rq * rb)).clamp(-1.0, 1.0)
                        } else {
                            1.0
                        };
                        clairaut_distance(self.profile, rq, rb, cos_t)
                    }
                    _ => unreachable!(),
                }
            };
            let eff_b = if matches!(self.method, Tier3Method::ExactClairaut) {
                ef.max(budget)
            } else {
                budget
            };
            return search_graph_with_rerank(
                SearchRequest {
                    graph: &self.graph,
                    seeds: &self.graph.entry_points,
                    ef,
                    budget: eff_b,
                    top_k,
                    n: self.n,
                },
                None::<&fn(usize) -> f64>,
                None::<&fn(usize, f64) -> bool>,
                &nop_prefetch,
                &beam,
                &exact_dist,
            );
        }

        // 复用通用残差锥/扁平方向编码与 L1 查询切片预备
        let st = self.store.as_ref().unwrap();
        let q_state = QueryDirState::from_vector(query, rq, self.cones.as_ref());

        let fixed_curvature = match self.method {
            Tier3Method::FixedConstantHyperbolicSm { curvature } => curvature,
            _ => 1.0,
        };
        let fixed_root = fixed_curvature.sqrt();
        let rq_ball = (0.5 * fixed_root * rq).tanh() / fixed_root;
        let q_bf_fixed = (1.0 - fixed_curvature * rq_ball * rq_ball)
            .max(1e-12)
            .recip();

        let q_slice = self.table.as_ref().map(|t| t.slice_for_query(rq));

        let eval_proxy = |node: usize, cos_hat: f64| -> f64 {
            match self.method {
                Tier3Method::FixedConstantHyperbolicSm { .. } => {
                    let m = self.sm_metas[node];
                    let rb = f64::from(m.r_ball);
                    let dr = rq_ball - rb;
                    (dr * dr + 2.0 * rq_ball * rb * (1.0 - cos_hat)).max(0.0)
                        * q_bf_fixed
                        * f64::from(m.bf_fixed)
                }
                Tier3Method::LocalWarpedFormulaSm => {
                    let rb = f64::from(self.beam_metas[node].r);
                    let dr = rq - rb;
                    let phi_mid = self.profile.phi(0.5 * (rq + rb));
                    dr * dr + 2.0 * phi_mid * phi_mid * (1.0 - cos_hat).max(0.0)
                }
                Tier3Method::WarpedTableSm => q_slice
                    .as_ref()
                    .unwrap()
                    .eval(self.beam_metas[node].r, cos_hat),
                _ => unreachable!(),
            }
        };

        let beam = |node: usize| -> f64 {
            let bm = self.beam_metas[node];
            let sm = self.sm_metas[node].sm_params();
            let (pp, sp) = st.node_ptrs(node);
            let cos_hat = q_state.eval_beam_cos(
                bm.cone_id as usize,
                f64::from(bm.cone_alpha),
                f64::from(bm.cone_beta),
                sm,
                pp,
                sp,
                self.dim,
            );
            eval_proxy(node, cos_hat)
        };
        let stage05 = |node: usize| -> f64 {
            let bm = self.beam_metas[node];
            let sm = self.sm_metas[node].sm_params();
            let cos_hat = q_state.eval_stage05_cos(
                bm.cone_id as usize,
                f64::from(bm.cone_alpha),
                f64::from(bm.cone_beta),
                sm,
                st.pos_slice(node),
                st.strong_slice(node),
            );
            eval_proxy(node, cos_hat)
        };
        let prune_lb = |node: usize, bound: f64| -> bool {
            let bm = self.beam_metas[node];
            let rv = f64::from(bm.r);
            let dr = (rq - rv).abs();
            if dr - 1e-4 > bound {
                return true;
            }
            // 由 φ 单调递增 + 测地近点下界 R_min = max(0, (R_q + R_v - bound)/2) 与锥角 Cauchy-Schwarz 上界得严格距离下界
            if let Some(cos_max) = q_state.cos_upper_bound(
                bm.cone_id as usize,
                f64::from(bm.cone_alpha),
                f64::from(bm.cone_beta),
            ) && cos_max < 1.0 - 1e-4
            {
                let r_min = ((rq + rv - bound) * 0.5).max(0.0);
                let phi_min = self.profile.phi(r_min);
                let ang_sq = 2.0 * (1.0 - cos_max) * phi_min * phi_min;
                let lb = (dr * dr + ang_sq).sqrt();
                if lb - 1e-4 > bound {
                    return true;
                }
            }
            false
        };
        let prefetch = |node: usize| self.prefetch_beam_node(node);

        search_graph_with_rerank(
            SearchRequest {
                graph: &self.graph,
                seeds: &self.graph.entry_points,
                ef,
                budget,
                top_k,
                n: self.n,
            },
            Some(&stage05),
            if use_radial_bound && matches!(self.method, Tier3Method::WarpedTableSm) {
                Some(&prune_lb)
            } else {
                None
            },
            &prefetch,
            &beam,
            &exact_dist,
        )
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  50k 纯合成流形生成器 + 精确真值计算器
// ═════════════════════════════════════════════════════════════════════════════

/// 生成混合/乘积流形合成数据集（每个子块内部含异构聚簇与径向梯度分布）。
pub fn generate_product_synthetic(
    factors: &[ProductFactor],
    n_docs: usize,
    n_queries: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let total_dim: usize = factors.iter().map(|f| f.dim()).sum();
    let mut rng = DetRng::new(seed);
    let n_clusters = 32usize;

    // 为每个因子各自准备 32 个聚类方向中心
    let centers: Vec<Vec<Vec<f64>>> = factors
        .iter()
        .map(|f| {
            (0..n_clusters)
                .map(|_| {
                    let g: Vec<f64> = (0..f.dim()).map(|_| rng.gauss()).collect();
                    normalize_dir(&g)
                })
                .collect()
        })
        .collect();

    let mut docs = Vec::with_capacity(n_docs * total_dim);
    let mut queries = Vec::with_capacity(n_queries * total_dim);

    for is_query in [false, true] {
        let count = if is_query { n_queries } else { n_docs };
        let dst = if is_query { &mut queries } else { &mut docs };
        for i in 0..count {
            // 为保证异构乘积困难度：不同因子独立选择聚类 id，制造真实多块耦合近邻竞争
            let cid_base = (i * 17 + usize::from(is_query) * 5) % n_clusters;
            for (bi, &f) in factors.iter().enumerate() {
                let cid = (cid_base + bi * 7 + (rng.next_u64() as usize % 3)) % n_clusters;
                let center = &centers[bi][cid];
                let mut dir: Vec<f64> = center
                    .iter()
                    .map(|&c| c + 0.35 * rng.gauss() / (f.dim() as f64).sqrt())
                    .collect();
                dir = normalize_dir(&dir);

                let scale = match f {
                    ProductFactor::Hyperbolic { curvature, .. } => {
                        let depth = 0.6 + 3.4 * rng.uniform01();
                        let s = curvature.sqrt();
                        (depth * s * 0.5).tanh() / s
                    }
                    ProductFactor::Euclidean { .. } => 0.4 + 2.6 * rng.uniform01(),
                    ProductFactor::Spherical { curvature, .. } => 1.0 / curvature.sqrt(),
                };
                for x in dir {
                    dst.push((x * scale) as f32);
                }
            }
        }
    }
    (docs, queries)
}

/// 生成径向变曲率翘曲乘积合成数据集（支持聚簇多壳与均匀跨壳分布）。
pub fn generate_warped_synthetic(
    dim: usize,
    n_docs: usize,
    n_queries: usize,
    r_lo: f64,
    r_hi: f64,
    clustered: bool,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut rng = DetRng::new(seed);
    let n_clusters = 32usize;
    let centers: Vec<Vec<f64>> = (0..n_clusters)
        .map(|_| {
            let g: Vec<f64> = (0..dim).map(|_| rng.gauss()).collect();
            normalize_dir(&g)
        })
        .collect();

    let mut docs = Vec::with_capacity(n_docs * dim);
    let mut queries = Vec::with_capacity(n_queries * dim);

    for is_query in [false, true] {
        let count = if is_query { n_queries } else { n_docs };
        let dst = if is_query { &mut queries } else { &mut docs };
        for i in 0..count {
            let dir = if clustered {
                let cid = (i * 19 + usize::from(is_query) * 3) % n_clusters;
                let sigma = 0.25 / (dim as f64).sqrt();
                let raw: Vec<f64> = centers[cid]
                    .iter()
                    .map(|&c| c + sigma * rng.gauss())
                    .collect();
                normalize_dir(&raw)
            } else {
                let raw: Vec<f64> = (0..dim).map(|_| rng.gauss()).collect();
                normalize_dir(&raw)
            };
            // 覆盖从核心区到深外层的宽径向分布，充分暴露曲率 K(R) 随 R 的剧烈变化
            let u = rng.uniform01();
            let r = r_lo + (r_hi - r_lo) * u.powf(0.85);
            for x in dir {
                dst.push((x * r) as f32);
            }
        }
    }
    (docs, queries)
}

/// 并行计算乘积流形精确 Top-K 真值。
pub fn product_ground_truth(
    factors: &[ProductFactor],
    docs: &[f32],
    queries: &[f32],
    top_k: usize,
) -> Vec<Vec<usize>> {
    let total_dim: usize = factors.iter().map(|f| f.dim()).sum();
    let n_docs = docs.len() / total_dim;
    queries
        .par_chunks_exact(total_dim)
        .map(|q| {
            let mut heap: BinaryHeap<(OrdDist, usize)> = BinaryHeap::with_capacity(top_k + 1);
            for i in 0..n_docs {
                let d = exact_product_distance_sq(
                    factors,
                    q,
                    &docs[i * total_dim..(i + 1) * total_dim],
                );
                let entry = (OrdDist(d), i);
                if heap.len() < top_k {
                    heap.push(entry);
                } else if entry < *heap.peek().unwrap() {
                    heap.pop();
                    heap.push(entry);
                }
            }
            let mut v = heap.into_vec();
            v.sort_unstable();
            v.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

/// 并行计算 Tier-3 翘曲乘积精确 Top-K 真值（利用严格几何下界 `|R_q - R_v| <= d(q, v)` 加速，数学保证与全扫零误差）。
pub fn warped_ground_truth(
    profile: WarpedProfile,
    dim: usize,
    docs: &[f32],
    queries: &[f32],
    top_k: usize,
) -> Vec<Vec<usize>> {
    let n_docs = docs.len() / dim;
    let radii: Vec<f64> = docs
        .par_chunks_exact(dim)
        .map(|r| norm_squared_f64(r).sqrt())
        .collect();
    let tbl = WarpedProxyTable::build(
        profile,
        radii.iter().copied().fold(0.0, f64::max) + 1e-3,
        48,
        33,
    );

    queries
        .par_chunks_exact(dim)
        .map(|q| {
            let rq = norm_squared_f64(q).sqrt();
            // 先按快速启发式（局部表）挑前 top_k 个打底收紧上界，再按精确 |R_q - R_v| 下界全扫描剪枝
            let mut coarse: Vec<(OrdDist, usize, f64)> = Vec::with_capacity(n_docs);
            for i in 0..n_docs {
                let rv = radii[i];
                let row = &docs[i * dim..(i + 1) * dim];
                let cos_t = if rq > 1e-12 && rv > 1e-12 {
                    (dot_f32_f64(q, row) / (rq * rv)).clamp(-1.0, 1.0)
                } else {
                    1.0
                };
                let est = tbl.eval(rq, rv, cos_t);
                coarse.push((OrdDist(est), i, cos_t));
            }
            coarse.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

            let mut heap: BinaryHeap<(OrdDist, usize)> = BinaryHeap::with_capacity(top_k + 1);
            for &(_, idx, cos_t) in &coarse {
                let rv = radii[idx];
                if heap.len() >= top_k && (rq - rv).abs() > heap.peek().unwrap().0.0 {
                    continue;
                }
                let exact = clairaut_distance(profile, rq, rv, cos_t);
                let entry = (OrdDist(exact), idx);
                if heap.len() < top_k {
                    heap.push(entry);
                } else if entry < *heap.peek().unwrap() {
                    heap.pop();
                    heap.push(entry);
                }
            }
            let mut v = heap.into_vec();
            v.sort_unstable();
            v.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clairaut求解器与常曲率poincare闭式解严格一致() {
        for curvature in [0.25, 1.0, 4.0] {
            let profile = WarpedProfile::ConstantHyperbolic { curvature };
            let s = curvature.sqrt();
            for &rq in &[0.0, 0.25, 1.5, 3.2] {
                for &rv in &[0.0, 0.35, 1.5, 2.8] {
                    for &cos_t in &[1.0, 0.92, 0.3, -0.4, -0.95, -1.0] {
                        let got = clairaut_distance(profile, rq, rv, cos_t);
                        let br_q = (s * rq * 0.5).tanh() / s;
                        let br_v = (s * rv * 0.5).tanh() / s;
                        let delta_sq = br_q * br_q + br_v * br_v - 2.0 * br_q * br_v * cos_t;
                        let fq = 1.0 / (1.0 - curvature * br_q * br_q);
                        let fv = 1.0 / (1.0 - curvature * br_v * br_v);
                        let expected = poincare_distance(delta_sq * fq * fv, curvature);
                        assert!(
                            (got - expected).abs() <= 1e-10 * expected.max(1.0),
                            "c={curvature}, rq={rq}, rv={rv}, cos={cos_t}: got={got}, expected={expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn 变曲率剖面三角不等式对称性与查表精度有效() {
        for profile in [
            WarpedProfile::Softening { alpha: 0.5 },
            WarpedProfile::Steepening { alpha: 0.22 },
        ] {
            let tbl = WarpedProxyTable::build(profile, 4.0, 48, 33);
            let pts = [(0.4f64, 0.9f64), (2.3, -0.4), (3.1, 0.2)];
            let d01 = clairaut_distance(profile, pts[0].0, pts[1].0, pts[0].1);
            let d10 = clairaut_distance(profile, pts[1].0, pts[0].0, pts[0].1);
            assert!((d01 - d10).abs() < 1e-12);
            let d02 = clairaut_distance(profile, pts[0].0, pts[2].0, -0.7);
            let d12 = clairaut_distance(profile, pts[1].0, pts[2].0, 0.15);
            assert!(d02 <= d01 + d12 + 1e-10);
            assert!(d01 >= (pts[0].0 - pts[1].0).abs() - 1e-12);

            let sl = tbl.slice_for_query(1.7);
            let exact = clairaut_distance(profile, 1.7, 2.85, 0.6);
            let approx = sl.eval(2.85, 0.6).sqrt();
            assert!(
                (approx - exact).abs() / exact.max(1e-6) < 0.02,
                "查表近似误差过大: approx={approx}, exact={exact}"
            );
        }
    }

    #[test]
    fn 小规模乘积与翘曲检索流程跑通且高召回() {
        let factors = vec![
            ProductFactor::Hyperbolic {
                dim: 16,
                curvature: 4.0,
            },
            ProductFactor::Euclidean { dim: 16 },
            ProductFactor::Spherical {
                dim: 16,
                curvature: 1.0,
            },
        ];
        let (docs, queries) = generate_product_synthetic(&factors, 400, 20, 77);
        let truth = product_ground_truth(&factors, &docs, &queries, 5);
        let idx = ProductBlockSmIndex::build(factors, &docs, 16, 48, 1.2, true);
        let mut hits = 0usize;
        for (qi, q) in queries.as_chunks::<48>().0.iter().enumerate() {
            let out = idx.search(q, 5, 48, 48, true, true);
            assert_eq!(out.results.len(), 5);
            hits += out
                .results
                .iter()
                .filter(|(id, _)| truth[qi].contains(id))
                .count();
        }
        assert!(hits as f64 / (20.0 * 5.0) >= 0.90);

        let profile = WarpedProfile::Steepening { alpha: 0.22 };
        let (wdocs, wqueries) = generate_warped_synthetic(32, 400, 20, 0.2, 3.2, true, 88);
        let wtruth = warped_ground_truth(profile, 32, &wdocs, &wqueries, 5);
        let widx = WarpedManifoldIndex::build(
            profile,
            Tier3Method::WarpedTableSm,
            32,
            &wdocs,
            16,
            48,
            1.2,
        );
        let mut whits = 0usize;
        for (qi, q) in wqueries.as_chunks::<32>().0.iter().enumerate() {
            let out = widx.search(q, 5, 48, 48, true);
            assert_eq!(out.results.len(), 5);
            whits += out
                .results
                .iter()
                .filter(|(id, _)| wtruth[qi].contains(id))
                .count();
        }
        assert!(whits as f64 / (20.0 * 5.0) >= 0.90);
    }
}
