"""在固定清洗数据上逐项修复读出方法，并延后读取开发集评分。"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

import run_low_capacity_experiment as old
from audit_readout_splits import OUTPUT
from diagnose_readout import captured_score, load_data, save_json, distribution, add_pilot_metric


VARIANTS = {
    "R0_control": {"calibrated": False, "aligned": False, "loss": "any", "shared": False, "squared_h1": False},
    "R1_scale": {"calibrated": True, "aligned": False, "loss": "any", "shared": False, "squared_h1": False},
    "R2_aligned": {"calibrated": True, "aligned": True, "loss": "any", "shared": False, "squared_h1": False},
    "R3_all_positive": {"calibrated": True, "aligned": True, "loss": "all", "shared": False, "squared_h1": False},
    "R4_shared": {"calibrated": True, "aligned": True, "loss": "all", "shared": True, "squared_h1": False},
    "R5_squared_h1": {"calibrated": True, "aligned": True, "loss": "all", "shared": True, "squared_h1": True},
}
MODELS = {name.split("_")[0]: cls for name, cls in old.MODEL_CLASSES.items() if not name.startswith("B0")}


class RepairedScorer(nn.Module):
    def __init__(self, name, input_dim, config, output_dim=32):
        super().__init__()
        self.name, self.config = name, config
        self.inner = MODELS[name](input_dim, output_dim)
        self.register_buffer("scale", torch.ones(()))
        if name != "B1" and config["aligned"]:
            self.inner.document_projection.load_state_dict(self.inner.query_projection.state_dict())
            self.inner.document_radius.load_state_dict(self.inner.query_radius.state_dict())
        if name != "B1" and config["shared"]:
            self.inner.document_projection = self.inner.query_projection
            self.inner.document_radius = self.inner.query_radius

    def parts(self, query, documents):
        _, geometric = captured_score(self.inner, query, documents)
        if self.name == "H1" and self.config["squared_h1"]:
            geometric = -(geometric ** 2)
        if self.config["calibrated"]:
            ratio = (-geometric).clamp_min(0) / self.scale
            residual = -ratio / (1 + ratio)
            # 相对无量纲距离的导数，用于追踪是否仍出现梯度压缩。
            derivative = 1 / (1 + ratio) ** 2
        else:
            residual = torch.tanh(geometric)
            derivative = 1 - residual ** 2
        base = self.inner.base_score(query, documents)
        weight = torch.sigmoid(self.inner.residual_logit)
        score = (base + weight * residual) / old.TEMPERATURE
        return score, geometric, residual, derivative

    def score(self, query, documents):
        return self.parts(query, documents)[0]

    @torch.no_grad()
    def calibrate(self, train):
        # 只用初始化模型与训练候选的距离，不使用任何相关性标签。
        distances = []
        for example in train["examples"]:
            q, d, _ = tensors(train, example)
            _, geometric = captured_score(self.inner, q, d)
            distance = -geometric
            if self.name == "H1" and self.config["squared_h1"]:
                distance = distance ** 2
            distances.append(distance)
        self.scale.copy_(torch.cat(distances).median().clamp_min(1e-8))


def tensors(split, example):
    return (split["tensor_queries"][example["query_index"]],
            split["tensor_documents"][example["candidate_indices"]],
            torch.from_numpy(example["positive_mask"]))


def retrieval_loss(scores, mask, mode):
    if not mask.any():
        raise ValueError("训练候选缺少正例")
    if mode == "any":
        return old.multi_positive_loss(scores, mask)
    # 每个正例都承担损失，单个高分正例不能替其他正例掩盖低分。
    return torch.logsumexp(scores, dim=0) - scores[mask].mean()


def ranking_metrics(scores, mask):
    order = torch.argsort(scores.detach(), descending=True).cpu().numpy()
    positives = np.asarray(mask)[order]
    positive_positions = np.flatnonzero(positives) + 1
    if not len(positive_positions):
        raise ValueError("评估候选缺少正例")
    return {"mrr": float(1 / positive_positions[0]),
            "map": float(np.mean(np.arange(1, len(positive_positions) + 1) / positive_positions)),
            "recall@10": float(np.mean(positive_positions <= 10)),
            "evidence_set_recall@10": float(np.all(positive_positions <= 10)),
            "perfect_positive_order": float(positive_positions[-1] == len(positive_positions))}, order


def evaluate(model, split, mode="all", baseline=False):
    rows = []
    with torch.no_grad():
        for example in split["examples"]:
            q, d, mask = tensors(split, example)
            scores = old.FixedWhitenedCosine().score(q, d) if baseline else model.score(q, d)
            if not torch.isfinite(scores).all():
                raise ValueError("检索分数出现非有限值")
            row, order = ranking_metrics(scores, mask)
            row["loss"] = float(retrieval_loss(scores, mask, mode))
            add_pilot_metric(row, order, example)
            rows.append({"query_id": example["query_id"], **row})
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "query_id"}
    return {"summary": summary, "per_query": rows}


def better(candidate, incumbent):
    # MAP 关注全部证据；相同 MAP 时比较 MRR，完全相同时保留更早状态/B0。
    for key in ("map", "mrr"):
        if candidate[key] > incumbent[key] + 1e-8:
            return True
        if candidate[key] < incumbent[key] - 1e-8:
            return False
    return False


def monitor(model, train, mode):
    raw, residual, derivative, norms = [], [], [], []
    for example in train["examples"]:
        q, d, mask = tensors(train, example)
        scores, geometric, bounded, local_derivative = model.parts(q, d)
        loss = retrieval_loss(scores, mask, mode)
        grads = torch.autograd.grad(loss, tuple(model.parameters()))
        if any(not torch.isfinite(grad).all() for grad in grads):
            raise ValueError("模型梯度出现非有限值")
        norms.append(float(torch.sqrt(sum(torch.sum(grad ** 2) for grad in grads))))
        raw.extend(geometric.detach().tolist())
        residual.extend(bounded.detach().tolist())
        derivative.extend(local_derivative.detach().tolist())
    values = np.array(derivative)
    result = {"raw": distribution(raw), "residual": distribution(residual),
              "derivative": distribution(derivative), "derivative_exact_zero": float(np.mean(values == 0)),
              "derivative_lt_1e_3": float(np.mean(values < 1e-3)),
              "per_query_parameter_gradient_l2": distribution(norms),
              "scale": float(model.scale), "weight": float(torch.sigmoid(model.inner.residual_logit).detach())}
    if model.name != "B1":
        with torch.no_grad():
            activations = torch.cat((model.inner.query_radius(train["tensor_queries"]).flatten(),
                                     model.inner.document_radius(train["tensor_documents"]).flatten()))
            probabilities = torch.sigmoid(activations)
            result["radius_sigmoid_derivative_lt_1e_5"] = float(((probabilities * (1 - probabilities)) < 1e-5).float().mean())
    return result


def prepare_data(audit):
    datasets, _, _ = load_data()
    for name in ("train", "validation"):
        split = datasets[name]
        rows = audit["training_metadata" if name == "train" else "validation_metadata"]
        ids = json.loads((old.EMBEDDING_DIR / name / "document_ids.json").read_text(encoding="utf-8"))
        positions = {doc_id: i for i, doc_id in enumerate(ids)}
        query_ids = json.loads((old.EMBEDDING_DIR / name / "query_ids.json").read_text(encoding="utf-8"))
        qpositions = {qid: i for i, qid in enumerate(query_ids)}
        kept_ids = sorted({doc_id for row in rows for doc_id in row["candidate_ids"]})
        doc_positions = {doc_id: i for i, doc_id in enumerate(kept_ids)}
        split["raw_documents"] = split["raw_documents"][[positions[doc_id] for doc_id in kept_ids]]
        split["raw_queries"] = split["raw_queries"][[qpositions[row["query_id"]] for row in rows]]
        split["examples"] = [{"query_id": row["query_id"], "query_index": i,
                              "candidate_indices": [doc_positions[doc_id] for doc_id in row["candidate_ids"]],
                              "candidate_ids": row["candidate_ids"],
                              "positive_mask": np.array([doc_id in row["positive_ids"] for doc_id in row["candidate_ids"]])}
                             for i, row in enumerate(rows)]
    mean, std = old.fit_whitening(datasets["train"]["raw_documents"])
    for split in datasets.values():
        for kind in ("documents", "queries"):
            split[kind] = old.apply_whitening(split["raw_" + kind], mean, std)
            split["tensor_" + kind] = torch.from_numpy(split[kind])
    return datasets, mean, std


def train_one(name, variant, seed, datasets, output):
    config = VARIANTS[variant]
    torch.manual_seed(seed)
    model = RepairedScorer(name, datasets["train"]["documents"].shape[1], config)
    model.calibrate(datasets["train"])
    initial_state = copy.deepcopy(model.state_dict())
    initial_monitor = monitor(model, datasets["train"], config["loss"])
    baseline = evaluate(model, datasets["validation"], config["loss"], baseline=True)
    best = baseline["summary"]
    best_state = copy.deepcopy(initial_state)
    selected_kind, selected_epoch = "B0", None
    initial_validation = evaluate(model, datasets["validation"], config["loss"])
    if better(initial_validation["summary"], best):
        best = initial_validation["summary"]
        selected_kind, selected_epoch = "residual", 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=old.LEARNING_RATE, weight_decay=old.WEIGHT_DECAY)
    rng = np.random.RandomState(seed)
    history = [{"epoch": 0, "validation": initial_validation["summary"], "train": evaluate(model, datasets["train"], config["loss"])["summary"]}]
    stale = 0
    for epoch in range(1, old.EPOCHS + 1):
        order = rng.permutation(len(datasets["train"]["examples"]))
        batch_gradients = []
        for start in range(0, len(order), old.BATCH_QUERIES):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in order[start:start + old.BATCH_QUERIES]:
                q, d, mask = tensors(datasets["train"], datasets["train"]["examples"][int(index)])
                losses.append(retrieval_loss(model.score(q, d), mask, config["loss"]))
            loss = torch.stack(losses).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            batch_gradients.append(float(norm))
            optimizer.step()
        validation = evaluate(model, datasets["validation"], config["loss"])
        history.append({"epoch": epoch, "validation": validation["summary"],
                        "train": evaluate(model, datasets["train"], config["loss"])["summary"],
                        "batch_gradient_l2_before_clip": distribution(batch_gradients)})
        if better(validation["summary"], best):
            best = validation["summary"]
            best_state = copy.deepcopy(model.state_dict())
            selected_kind, selected_epoch, stale = "residual", epoch, 0
        else:
            stale += 1
        if stale >= old.PATIENCE:
            break
    last_monitor = monitor(model, datasets["train"], config["loss"])
    last_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    selected_baseline = selected_kind == "B0"
    checkpoint = output / f"{variant}_{name}_seed{seed}.pt"
    torch.save({"initial": initial_state, "selected": best_state, "last": last_state,
                "selected_kind": selected_kind, "selected_epoch": selected_epoch}, checkpoint)
    result = {"variant": variant, "model": name, "seed": seed, "config": config,
              "parameters": old.parameter_count(model), "selected_kind": selected_kind, "selected_epoch": selected_epoch,
              "baseline_validation": baseline, "selected_validation": evaluate(model, datasets["validation"], config["loss"], selected_baseline),
              "selected_train": evaluate(model, datasets["train"], config["loss"], selected_baseline),
              "history": history, "initial_monitor": initial_monitor, "last_monitor": last_monitor,
              "checkpoint": checkpoint.name, "checkpoint_sha256": old.sha256_file(checkpoint)}
    save_json(output / f"{variant}_{name}_seed{seed}.json", result)
    return result


def small_fit(name, train):
    # 仅训练前4题，使用几何分支独立评分，排除高维基线和小门控掩盖拟合能力。
    torch.manual_seed(11)
    config = VARIANTS["R5_squared_h1" if name == "H1" else "R4_shared"]
    model = RepairedScorer(name, train["documents"].shape[1], config)
    model.calibrate(train)
    examples = sorted(train["examples"], key=lambda row: row["query_id"])[:4]
    optimizer = torch.optim.AdamW(model.parameters(), lr=old.LEARNING_RATE, weight_decay=old.WEIGHT_DECAY)
    history = []
    for step in range(201):
        losses, metrics = [], []
        for example in examples:
            q, d, mask = tensors(train, example)
            residual = model.parts(q, d)[2]
            score = residual / old.TEMPERATURE
            losses.append(retrieval_loss(score, mask, "all"))
            metrics.append(ranking_metrics(score, mask)[0])
        loss = torch.stack(losses).mean()
        if step in (0, 1, 10, 50, 100, 200):
            history.append({"step": step, "loss": float(loss.detach()),
                            "map": float(np.mean([row["map"] for row in metrics])),
                            "perfect_positive_order": float(np.mean([row["perfect_positive_order"] for row in metrics]))})
        if step == 200:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    return {"model": name, "query_ids": [row["query_id"] for row in examples], "steps": 200,
            "protocol": "仅训练4题，几何分支独立评分，固定温度0.1；无验证选模，无开发评分。此测试不代表正式残差模型泛化。",
            "history": history}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    audit = json.loads((OUTPUT / "split_audit.json").read_text(encoding="utf-8"))
    datasets, mean, std = prepare_data(audit)
    input_paths = [Path(__file__), Path(old.__file__), Path(__file__).with_name("audit_readout_splits.py"),
                   Path(__file__).with_name("diagnose_readout.py"), OUTPUT / "split_audit.json"]
    for name in ("train", "validation"):
        input_paths += [old.EMBEDDING_DIR / name / filename for filename in ("documents.npy", "queries.npy", "document_ids.json", "query_ids.json")]
    protocol = {"variants": VARIANTS, "seeds": list(old.SEEDS), "epochs": old.EPOCHS, "patience": old.PATIENCE,
                "selection": "所有版本统一按验证MAP、其次MRR选模。B0先入选；只有严格提高才替换；完全并列保留B0/更早状态。",
                "calibration": "尺度为初始化模型的训练候选非负距离中位数；冻结后使用 -x/(1+x)，x=距离/尺度。不使用留出集拟合，不依赖候选集合进行在线归一化。",
                "data": {name: {"queries": len(split["examples"]), "documents": len(split["documents"])} for name, split in datasets.items()},
                "scope": "主对照为R4的B1/B2/H2和R5的H1，共享映射且采用平方距离；其他版本为预先指定的逐项消融。高维B0始终保留，32维只用于残差。",
                "limitations": ["仍是14题开发集，不能据此宣称泛化。", "验证候选未扩展到完整Haystack。",
                                "跨集合语义近重复与未标注假负例尚未人工穷举审计。", "B2/H1/H2同容量；B1缺少半径头，单列为较小基线。",
                                "R0只复用旧模型定义，清洗数据与MAP/B0选模协议已改变，不能直接与旧V3做单因素比较。",
                                "共享映射得到对称且候选集合无关的评分，但该残差和平方距离组合并不自动成为一般黎曼测地度量。"],
                "input_sha256": {str(path.relative_to(old.ROOT)): old.sha256_file(path) for path in input_paths}}
    save_json(OUTPUT / "protocol.json", protocol)
    np.save(OUTPUT / "whitening_mean.npy", mean)
    np.save(OUTPUT / "whitening_std.npy", std)
    results = []
    with threadpool_limits(limits=1):
        for variant in VARIANTS:
            names = ("H1",) if variant == "R5_squared_h1" else tuple(MODELS)
            # B1 本身没有双塔，R2等同R1，R4等同R3，复用结果而不制造重复训练。
            if variant in ("R2_aligned", "R4_shared"):
                names = ("B2", "H1", "H2")
            for name in names:
                for seed in old.SEEDS:
                    print(f"开始重测 {variant} / {name}，随机种子 {seed}", flush=True)
                    row = train_one(name, variant, seed, datasets, OUTPUT)
                    results.append(row)
                    print(f"已冻结选择：{row['selected_kind']}，轮次 {row['selected_epoch']}，验证MAP {row['selected_validation']['summary']['map']:.4f}", flush=True)
        registry = [{key: row[key] for key in ("variant", "model", "seed", "selected_kind", "selected_epoch", "checkpoint", "checkpoint_sha256")} for row in results]
        save_json(OUTPUT / "selection_registry.json", registry)
        print("全部模型选择已落盘，开始一次性开发集评估。", flush=True)
        baseline = {name: evaluate(None, split, baseline=True) for name, split in datasets.items()}
        for row in results:
            model = RepairedScorer(row["model"], datasets["train"]["documents"].shape[1], row["config"])
            checkpoint = torch.load(OUTPUT / row["checkpoint"], map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["selected"])
            row["development"] = evaluate(model, datasets["development"], row["config"]["loss"], row["selected_kind"] == "B0")
            row["initial_development"] = None
            save_json(OUTPUT / f"{row['variant']}_{row['model']}_seed{row['seed']}.json", row)
        fits = [small_fit(name, datasets["train"]) for name in MODELS]
    save_json(OUTPUT / "results.json", {"protocol": protocol, "baseline": baseline, "runs": results, "small_fit": fits,
                                        "selection_registry_sha256": old.sha256_file(OUTPUT / "selection_registry.json")})
    print("重测完成，已保存全部结果与4题拟合诊断。", flush=True)


if __name__ == "__main__":
    main()
