"""复现 V3 的可学习性诊断，并隔离降维与双塔初始化的影响。"""

import argparse
import copy
import json
import platform
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_low_capacity_experiment as experiment
from evaluate_cosine_baseline import ndcg_at_k


DEFAULT_OUTPUT = experiment.ROOT / "artifacts" / "diagnostics" / "readout_steps_1_2"


def distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not values.size or not np.isfinite(values).all():
        raise ValueError("统计输入为空或含非有限值")
    quantiles = np.quantile(values, [0, 0.01, 0.5, 0.99, 1])
    return dict(zip(("min", "p01", "median", "p99", "max"), map(float, quantiles))) | {
        "mean": float(values.mean()), "std": float(values.std()), "count": int(values.size)
    }


def ranking_diagnostics(scores, baseline, mask):
    # 与原训练器一致，使用 CPU torch.argsort 的并列分数处理方式。
    scores = torch.as_tensor(scores).detach().cpu()
    baseline = torch.as_tensor(baseline).detach().cpu()
    order = torch.argsort(scores, descending=True).numpy()
    base_order = torch.argsort(baseline, descending=True).numpy()
    ranks = np.empty(len(order), dtype=np.int64)
    base_ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order))
    base_ranks[base_order] = np.arange(len(order))
    positive_ranks = ranks[np.asarray(mask, dtype=bool)] + 1
    k = min(10, len(order))
    row = {
        "mrr": float(1 / positive_ranks.min()) if positive_ranks.size else None,
        "recall@10": float(np.mean(positive_ranks <= 10)) if positive_ranks.size else None,
        "evidence_set_recall@10": float(np.all(positive_ranks <= 10)) if positive_ranks.size else None,
        "ranking_changed": float(not np.array_equal(order, base_order)),
        "top1_changed": float(order[0] != base_order[0]),
        "top10_overlap": len(set(order[:k]) & set(base_order[:k])) / k,
        "rank_spearman": float(1 - 6 * np.sum((ranks - base_ranks) ** 2) / (len(order) * (len(order) ** 2 - 1)))
        if len(order) > 1 else 1.0,
        "tie_fraction": 1 - len(torch.unique(scores)) / len(scores),
        "positive_ranks": positive_ranks.tolist(),
    }
    return row, order


def aggregate_rankings(rows):
    return {
        key: float(np.mean([row[key] for row in rows if row.get(key) is not None]))
        for key in rows[0]
        if key not in ("query_id", "positive_ranks")
        and any(row.get(key) is not None for row in rows)
    }


def load_data():
    raw_train = np.load(experiment.EMBEDDING_DIR / "train" / "documents.npy")
    mean, std = experiment.fit_whitening(raw_train)
    datasets = {}
    for name in ("train", "validation"):
        split = experiment.load_split(name, mean, std)
        for kind in ("documents", "queries"):
            split["raw_" + kind] = np.load(experiment.EMBEDDING_DIR / name / f"{kind}.npy")
        metadata = experiment.read_jsonl(experiment.DATA_DIR / f"{name}_metadata.jsonl")
        for example, row in zip(split["examples"], metadata, strict=True):
            example["candidate_ids"] = row["candidate_ids"]
        datasets[name] = split

    root = experiment.TEST_EMBEDDING_DIR
    data_root = experiment.TEST_DATA_DIR
    document_ids = json.loads((root / "document_ids.json").read_text(encoding="utf-8"))
    query_ids = json.loads((root / "query_ids_instructed.json").read_text(encoding="utf-8"))
    doc_index = {value: i for i, value in enumerate(document_ids)}
    query_index = {value: i for i, value in enumerate(query_ids)}
    candidates = defaultdict(list)
    for row in experiment.read_jsonl(data_root / "documents.jsonl"):
        candidates[str(row["question_id"])].append(str(row["document_id"]))
    official = experiment.group_qrels(experiment.read_jsonl(data_root / "official_qrels.jsonl"))
    pilot = experiment.group_qrels(experiment.read_jsonl(data_root / "pilot_qrels.jsonl"))
    dev = {"examples": []}
    for kind, filename in (("documents", "documents.npy"), ("queries", "queries_instructed.npy")):
        dev["raw_" + kind] = np.load(root / filename)
        dev[kind] = experiment.apply_whitening(dev["raw_" + kind], mean, std)
    for row in experiment.read_jsonl(data_root / "queries.jsonl"):
        query_id = str(row["query_id"])
        ids = candidates[query_id]
        dev["examples"].append({
            "query_id": query_id, "query_index": query_index[query_id],
            "candidate_indices": [doc_index[item] for item in ids], "candidate_ids": ids,
            "positive_mask": np.array([item in official[query_id] for item in ids]),
            "pilot_relevance": {item: int(label["relevance"]) for item, label in pilot[query_id].items()},
        })
    datasets["development"] = dev
    for split in datasets.values():
        for kind in ("documents", "queries", "raw_documents", "raw_queries"):
            values = split[kind]
            if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
                raise ValueError("向量包含非有限值或零向量")
        split["tensor_documents"] = torch.from_numpy(split["documents"])
        split["tensor_queries"] = torch.from_numpy(split["queries"])
    return datasets, mean, std


def captured_score(model, query, documents):
    # 截取原模型传给 combine 的张量，不另写一套几何公式。
    original = model.combine
    captured = {}

    def intercept(self, q, d, geometric_score):
        captured["geometric"] = geometric_score
        return original(q, d, geometric_score)

    model.combine = types.MethodType(intercept, model)
    try:
        score = model.score(query, documents)
    finally:
        del model.combine
    return score, captured["geometric"]


def add_pilot_metric(row, order, example):
    if "pilot_relevance" in example:
        ids = [example["candidate_ids"][i] for i in order]
        row["pilot_ndcg@10"] = ndcg_at_k(ids, example["pilot_relevance"], 10)


def inspect_split(model, split, gradients=False):
    model.eval()
    parameters = dict(model.named_parameters())
    grad_norms = defaultdict(list)
    grad_nonzero = defaultdict(list)
    grad_sums = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
    raw_scores, bounded_scores, derivatives = [], [], []
    residual_spans, base_spans, losses, upstream, downstream = [], [], [], [], []
    rankings, geometry_rankings, per_query = [], [], []
    weight = float(torch.sigmoid(model.residual_logit).detach())
    for example in split["examples"]:
        query = split["tensor_queries"][example["query_index"]]
        documents = split["tensor_documents"][example["candidate_indices"]]
        with torch.set_grad_enabled(gradients):
            scores, geometric = captured_score(model, query, documents)
            bounded = torch.tanh(geometric)
            base = model.base_score(query, documents) / experiment.TEMPERATURE
            mask = torch.from_numpy(example["positive_mask"])
            loss = experiment.multi_positive_loss(scores, mask)
            if gradients:
                score_grad = torch.autograd.grad(loss, scores, retain_graph=True)[0]
                geometry_grad = torch.autograd.grad(loss, geometric, retain_graph=True)[0]
                values = torch.autograd.grad(loss, tuple(parameters.values()))
                for (name, parameter), grad in zip(parameters.items(), values, strict=True):
                    if not torch.isfinite(grad).all():
                        raise ValueError(f"参数梯度非有限：{name}")
                    grad_norms[name].append(float(torch.linalg.vector_norm(grad)))
                    grad_nonzero[name].append(float(torch.count_nonzero(grad) / parameter.numel()))
                    grad_sums[name].add_(grad.detach())
                upstream.append(float(torch.linalg.vector_norm(score_grad)))
                downstream.append(float(torch.linalg.vector_norm(geometry_grad)))
        raw = geometric.detach().numpy()
        bound = bounded.detach().numpy()
        raw_scores.extend(raw.tolist())
        bounded_scores.extend(bound.tolist())
        derivatives.extend((1 - bound ** 2).tolist())
        residual_span = float(np.ptp(weight * bound))
        residual_spans.append(residual_span)
        base_spans.append(float(torch.max(base.detach()) - torch.min(base.detach())) * experiment.TEMPERATURE)
        losses.append(float(loss.detach()))
        rank, order = ranking_diagnostics(scores, base, example["positive_mask"])
        add_pilot_metric(rank, order, example)
        geometry_rank, geometry_order = ranking_diagnostics(geometric, base, example["positive_mask"])
        add_pilot_metric(geometry_rank, geometry_order, example)
        rankings.append(rank)
        geometry_rankings.append(geometry_rank)
        per_query.append({
            "query_id": example["query_id"], **rank,
            "loss": float(loss.detach()), "residual_span": residual_span,
            "tanh_exact_saturation": float(np.mean(np.abs(bound) == 1)),
            "tanh_derivative_lt_1e_3": float(np.mean(1 - bound ** 2 < 1e-3)),
            "geometry_only": geometry_rank,
        })
    derivative = np.array(derivatives)
    result = {
        "raw_geometric_score": distribution(raw_scores),
        "bounded_geometric_score": distribution(bounded_scores),
        "tanh_derivative": distribution(derivatives),
        "tanh_exact_saturation": float(np.mean(derivative == 0)),
        "tanh_derivative_lt_1e_3": float(np.mean(derivative < 1e-3)),
        "residual_weight": weight, "residual_span": distribution(residual_spans),
        "base_span": distribution(base_spans), "loss_mean": float(np.mean(losses)),
        "ranking": aggregate_rankings(rankings),
        "geometry_only_ranking": aggregate_rankings(geometry_rankings),
        "per_query": per_query,
    }
    if gradients:
        result["gradients"] = {
            name: {
                "per_query_l2": distribution(grad_norms[name]),
                "per_query_nonzero_fraction": distribution(grad_nonzero[name]),
                "mean_loss_gradient_l2": float(torch.linalg.vector_norm(grad_sums[name] / len(rankings))),
            } for name in parameters
        }
        result["loss_to_combined_score_gradient_l2"] = distribution(upstream)
        result["loss_to_geometry_gradient_l2"] = distribution(downstream)
    return result


def parameter_changes(model, initial):
    return {name: {
        "delta_l2": float(torch.linalg.vector_norm(value - initial[name])),
        "relative_delta_l2": float(torch.linalg.vector_norm(value - initial[name]) / torch.linalg.vector_norm(initial[name]).clamp_min(1e-12)),
        "max_absolute_delta": float(torch.max(torch.abs(value - initial[name]))),
    } for name, value in model.state_dict().items()}


def snapshot(model, datasets, initial):
    return {
        "parameter_changes_from_initial": parameter_changes(model, initial),
        "splits": {name: inspect_split(model, split, gradients=(name == "train")) for name, split in datasets.items()},
    }


def replay_model(name, seed, datasets, saved_run):
    torch.manual_seed(seed)
    model = experiment.MODEL_CLASSES[name](datasets["train"]["documents"].shape[1])
    initial = copy.deepcopy(model.state_dict())
    stages = {"initial": snapshot(model, datasets, initial)}
    history = []
    last_state = None
    original_evaluate = experiment.evaluate_validation

    def observe(current, validation, device):
        nonlocal last_state
        mrr = original_evaluate(current, validation, device)
        diagnostic = inspect_split(current, datasets["train"], gradients=True)
        diagnostic.pop("per_query")
        history.append({"epoch": len(history), "validation_mrr": mrr, "train": diagnostic,
                        "parameter_changes_from_initial": parameter_changes(current, initial)})
        last_state = copy.deepcopy(current.state_dict())
        return mrr

    experiment.evaluate_validation = observe
    try:
        model, training = experiment.train_model(model, datasets["train"], datasets["validation"], seed, torch.device("cpu"))
    finally:
        experiment.evaluate_validation = original_evaluate
    selected_state = copy.deepcopy(model.state_dict())
    stages["replayed_selected"] = snapshot(model, datasets, initial)
    model.load_state_dict(last_state)
    stages["last_before_restore"] = snapshot(model, datasets, initial)
    checkpoint = experiment.OUTPUT_DIR / saved_run["checkpoint"]
    saved_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    max_difference = max(float(torch.max(torch.abs(selected_state[key] - saved_state[key]))) for key in selected_state)
    model.load_state_dict(saved_state)
    stages["saved_selected"] = snapshot(model, datasets, initial)
    return {
        "model": name, "seed": seed, "training": training, "epoch_diagnostics": history,
        "checkpoint_reproduction": {
            "path": str(checkpoint.relative_to(experiment.ROOT)),
            "sha256": experiment.sha256_file(checkpoint), "max_absolute_difference": max_difference,
            "bitwise_equal": all(torch.equal(selected_state[key], saved_state[key]) for key in selected_state),
            "saved_best_epoch": saved_run["training"]["best_epoch"],
            "replayed_best_epoch": training["best_epoch"],
            "saved_validation_history": saved_run["training"]["history"],
        }, "stages": stages,
    }


def evaluate_coordinates(split, documents, queries, baseline_documents, baseline_queries, score_type="cosine"):
    docs = torch.from_numpy(np.asarray(documents, dtype=np.float32))
    qs = torch.from_numpy(np.asarray(queries, dtype=np.float32))
    base_docs = experiment.normalize(torch.from_numpy(baseline_documents))
    base_qs = experiment.normalize(torch.from_numpy(baseline_queries))
    if score_type == "cosine":
        docs, qs = experiment.normalize(docs), experiment.normalize(qs)
    rows = []
    for example in split["examples"]:
        indices, query_index = example["candidate_indices"], example["query_index"]
        if score_type == "cosine":
            scores = docs[indices] @ qs[query_index]
        else:
            scores = -torch.sum((docs[indices] - qs[query_index]) ** 2, dim=-1)
        base = base_docs[indices] @ base_qs[query_index]
        row, order = ranking_diagnostics(scores, base, example["positive_mask"])
        add_pilot_metric(row, order, example)
        rows.append({"query_id": example["query_id"], **row})
    return {"summary": aggregate_rankings(rows), "per_query": rows}


def projection_controls(datasets, output):
    results = []
    for space in ("raw", "whitened"):
        prefix = "raw_" if space == "raw" else ""
        train_documents = datasets["train"][prefix + "documents"]
        # PCA 仅拟合训练文档，使用完整 SVD，避免随机近似额外引入变量。
        pca = PCA(n_components=experiment.OUTPUT_DIM, svd_solver="full")
        pca.fit(train_documents)
        np.savez(output / f"pca_{space}.npz", mean=pca.mean_, components=pca.components_,
                 explained_variance_ratio=pca.explained_variance_ratio_)
        print(f"已完成 {space} 空间的训练集 PCA，解释方差 {pca.explained_variance_ratio_.sum():.4f}", flush=True)
        methods = [("full_1536", None), ("train_centered_1536", None), ("train_pca_32", None)]
        methods += [(method, seed) for seed in experiment.SEEDS for method in ("shared_random_32", "independent_random_32")]
        for method, seed in methods:
            query_weight = document_weight = None
            if seed is not None:
                # 复用实际双塔的初始化顺序，两个条件保持相同 Query 权重。
                torch.manual_seed(seed)
                initialized = experiment.EuclideanTwoTower(train_documents.shape[1])
                query_weight = initialized.query_projection.weight.detach().numpy()
                document_weight = (query_weight if method == "shared_random_32"
                                   else initialized.document_projection.weight.detach().numpy())
            transformed = {}
            for split_name, split in datasets.items():
                documents = split[prefix + "documents"]
                queries = split[prefix + "queries"]
                if method == "train_centered_1536":
                    documents, queries = documents - pca.mean_, queries - pca.mean_
                elif method == "train_pca_32":
                    documents, queries = pca.transform(documents), pca.transform(queries)
                elif seed is not None:
                    documents, queries = documents @ document_weight.T, queries @ query_weight.T
                transformed[split_name] = (documents, queries)
            for score_type in ("cosine", "negative_squared_l2"):
                results.append({
                    "space": space, "method": method, "seed": seed, "score": score_type,
                    "pca_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()) if method == "train_pca_32" else None,
                    "splits": {split_name: evaluate_coordinates(
                        split, *transformed[split_name], split[prefix + "documents"], split[prefix + "queries"], score_type
                    ) for split_name, split in datasets.items()},
                })
    return results


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def input_hashes():
    paths = [Path(__file__), Path(experiment.__file__), Path(experiment.__file__).with_name("evaluate_cosine_baseline.py")]
    for name in ("train", "validation"):
        paths += [experiment.EMBEDDING_DIR / name / filename for filename in ("documents.npy", "queries.npy", "document_ids.json", "query_ids.json")]
        paths.append(experiment.DATA_DIR / f"{name}_metadata.jsonl")
    paths += [experiment.TEST_EMBEDDING_DIR / filename for filename in ("documents.npy", "queries_instructed.npy", "document_ids.json", "query_ids_instructed.json")]
    paths += [experiment.TEST_DATA_DIR / filename for filename in ("documents.jsonl", "queries.jsonl", "official_qrels.jsonl", "pilot_qrels.jsonl")]
    paths += [experiment.OUTPUT_DIR / "results.json", experiment.OUTPUT_DIR / "whitening_mean.npy", experiment.OUTPUT_DIR / "whitening_std.npy"]
    return {str(path.relative_to(experiment.ROOT)): experiment.sha256_file(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description="执行读出可学习性与投影诊断，不调用外部 API。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    datasets, mean, std = load_data()
    for filename, values in (("whitening_mean.npy", mean), ("whitening_std.npy", std)):
        if not np.array_equal(values, np.load(experiment.OUTPUT_DIR / filename)):
            raise ValueError("当前训练集统计与已保存 Whitening 不一致")
    saved = json.loads((experiment.OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    manifest = {
        "scope": "步骤1：V3 数值、梯度、参数与排序诊断；步骤2：无训练投影对照。",
        "splits": {name: {"queries": len(split["examples"]), "documents": len(split["documents"])} for name, split in datasets.items()},
        "limitations": [
            "现有14题仅作开发诊断，不能作为未接触的最终测试集。",
            "训练和验证仍使用现有32候选协议，本轮未执行步骤4的数据审计或扩充候选。",
            "PCA和Whitening仅拟合训练文档，但跨集合内容级隔离尚未核验。",
            "复现原有训练用于观察梯度，不增加小子集拟合实验，不修改原模型或选择协议。",
            "几何原始分数：B1/B2为负平方欧氏距离，H1为负双曲距离，H2为负混合平方距离。",
            "近饱和定义为tanh局部导数小于1e-3；精确饱和定义为float32局部导数等于0。",
            "梯度按每题计算；均值损失梯度另行汇总，均为裁剪前且不包含AdamW权重衰减。",
            "梯度仅使用训练标签；几何分支独立排名与投影均未重新训练。",
            "投影对照不含半径头；实际带半径的初始化几何评分另见步骤1。",
            "并列分数沿用当前CPU torch.argsort；另报告并列比例。",
        ],
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "torch_threads": 1},
        "seeds": list(experiment.SEEDS), "input_sha256": input_hashes(),
    }
    save_json(args.output / "manifest.json", manifest)
    with threadpool_limits(limits=1):
        projection = projection_controls(datasets, args.output)
        save_json(args.output / "projection_controls.json", projection)
        runs = []
        for name in experiment.MODEL_CLASSES:
            if name.startswith("B0"):
                continue
            for saved_run in saved["models"][name]["runs"]:
                seed = saved_run["seed"]
                print(f"开始诊断 {name}，随机种子 {seed}", flush=True)
                run = replay_model(name, seed, datasets, saved_run)
                save_json(args.output / f"{name}_seed{seed}.json", run)
                runs.append({"model": name, "seed": seed, "file": f"{name}_seed{seed}.json",
                             "checkpoint_reproduction": run["checkpoint_reproduction"]})
                print(f"已完成 {name}，最佳轮次 {run['training']['best_epoch']}，检查点最大差异 {run['checkpoint_reproduction']['max_absolute_difference']:.3g}", flush=True)
        save_json(args.output / "runs.json", runs)
    print(f"诊断产物已保存：{args.output}", flush=True)


if __name__ == "__main__":
    main()
