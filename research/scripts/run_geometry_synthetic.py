"""在已知欧氏/双曲几何上验证读出模型是否恢复真实测地关系。"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

import run_low_capacity_experiment as old


ROOT = old.ROOT
OUTPUT = ROOT / "artifacts" / "experiments" / "geometry_synthetic_v1"
SEEDS = (11, 23, 47, 71, 101)
DIMENSIONS = 2
TRAIN_SIZE = 2400
TEST_SIZE = 800
MAX_STEPS = 1200


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def euclidean_distance(x, y):
    return torch.linalg.vector_norm(x - y, dim=-1)


def poincare_distance(x, y):
    numerator = 2.0 * torch.sum((x - y) ** 2, dim=-1)
    denominator = (1.0 - torch.sum(x ** 2, dim=-1)) * (1.0 - torch.sum(y ** 2, dim=-1))
    return torch.acosh((1.0 + numerator / denominator).clamp_min(1.0 + 1e-7))


def sample_points(space, count, rng):
    angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
    if space == "euclidean":
        radius = rng.uniform(0.0, 1.0, size=count) ** 0.5
    elif space == "hyperbolic":
        # 在双曲半径0到2.2之间取样，使靠近边界的区域有足够质量。
        radius = np.tanh(rng.uniform(0.0, 2.2, size=count) / 2.0)
    else:
        raise ValueError(f"未知空间：{space}")
    return np.stack((radius * np.cos(angles), radius * np.sin(angles)), axis=1).astype(np.float32)


def make_dataset(space, size, seed):
    rng = np.random.default_rng(seed)
    qs, d1s, d2s, targets = [], [], [], []
    distance = euclidean_distance if space == "euclidean" else poincare_distance
    while len(qs) < size:
        points = sample_points(space, 3, rng)
        q, d1, d2 = [torch.from_numpy(row) for row in points]
        distances = distance(torch.stack((q, q)), torch.stack((d1, d2)))
        order = torch.argsort(distances)
        if float(distances[order[1]] - distances[order[0]]) < (0.16 if space == "euclidean" else 0.35):
            continue
        docs = (d1, d2) if int(order[0]) == 0 else (d2, d1)
        qs.append(q.numpy())
        d1s.append(docs[0].numpy())
        d2s.append(docs[1].numpy())
        targets.append(0)
    return {"q": torch.from_numpy(np.stack(qs)), "d1": torch.from_numpy(np.stack(d1s)),
            "d2": torch.from_numpy(np.stack(d2s)), "target": torch.tensor(targets, dtype=torch.long)}


class Readout(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == "shared_euclidean":
            self.map = nn.Linear(DIMENSIONS, DIMENSIONS, bias=False)
        elif name == "bilinear":
            self.qmap = nn.Linear(DIMENSIONS, 4, bias=False)
            self.dmap = nn.Linear(DIMENSIONS, 4, bias=False)

    def scores(self, q, docs):
        if self.name == "cosine":
            qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            dn = docs / docs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return (qn[:, None, :] * dn).sum(-1)
        if self.name == "euclidean_direct":
            return -torch.sum((q[:, None, :] - docs) ** 2, dim=-1)
        if self.name == "poincare_direct":
            return -poincare_distance(q[:, None, :].expand_as(docs), docs) ** 2
        if self.name == "shared_euclidean":
            return -torch.sum((self.map(q)[:, None, :] - self.map(docs)) ** 2, dim=-1)
        if self.name == "bilinear":
            return (self.qmap(q)[:, None, :] * self.dmap(docs)).sum(-1)
        raise ValueError(self.name)


def metrics(model, data):
    with torch.no_grad():
        docs = torch.stack((data["d1"], data["d2"]), dim=1)
        scores = model.scores(data["q"], docs)
        predictions = scores.argmax(-1)
        correct = predictions.eq(data["target"])
        return {"accuracy": float(correct.float().mean()), "loss": float((torch.logsumexp(scores, -1) - scores[:, 0]).mean())}


def train(name, data, seed):
    torch.manual_seed(seed)
    model = Readout(name)
    if name in ("cosine", "euclidean_direct", "poincare_direct"):
        return {"seed": seed, "name": name, "train": metrics(model, data), "test": None, "steps": 0}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    initial = metrics(model, data)
    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        docs = torch.stack((data["d1"], data["d2"]), dim=1)
        scores = model.scores(data["q"], docs)
        loss = (torch.logsumexp(scores, -1) - scores[:, 0]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0 and metrics(model, data)["accuracy"] >= 0.999:
            break
    return {"seed": seed, "name": name, "initial": initial, "train": metrics(model, data), "steps": step, "model": model}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = {"spaces": ["euclidean", "hyperbolic"],
                "models": ["cosine", "euclidean_direct", "poincare_direct", "shared_euclidean", "bilinear"],
                "seeds": SEEDS, "train_size": TRAIN_SIZE, "test_size": TEST_SIZE,
                "hyperbolic": "Poincare ball, curvature -1, distances use exact acosh formula",
                "success": "训练集按真实空间测地距离生成；测试集独立生成，不改变模型选择"}
    save_json(OUTPUT / "protocol.json", protocol)
    rows = []
    for space in protocol["spaces"]:
        train_data = make_dataset(space, TRAIN_SIZE, 20260918 + (0 if space == "euclidean" else 1000))
        test_data = make_dataset(space, TEST_SIZE, 20261918 + (0 if space == "euclidean" else 1000))
        np.savez(OUTPUT / f"{space}_data.npz", **{key: value.numpy() for key, value in train_data.items()},
                 test_q=test_data["q"].numpy(), test_d1=test_data["d1"].numpy(), test_d2=test_data["d2"].numpy(),
                 test_target=test_data["target"].numpy())
        for name in protocol["models"]:
            if name in ("cosine", "euclidean_direct", "poincare_direct"):
                row = train(name, train_data, 0)
                model = Readout(name)
                row["test"] = metrics(model, test_data)
                row["space"] = space
                rows.append(row)
                print(f"{space}，{name}：测试准确率{row['test']['accuracy']:.1%}", flush=True)
            else:
                for seed in SEEDS:
                    row = train(name, train_data, seed)
                    model = row.pop("model")
                    row["test"] = metrics(model, test_data)
                    row["space"] = space
                    rows.append(row)
                    print(f"{space}，{name}，种子{seed}：训练{row['train']['accuracy']:.1%}，测试{row['test']['accuracy']:.1%}，{row['steps']}步", flush=True)
    save_json(OUTPUT / "results.json", {"protocol": protocol, "runs": rows})
    lines = ["# 已知几何合成基准", "", "标签由真实空间的测地距离生成；训练/测试点独立。direct模型不训练，shared_euclidean与bilinear在训练集上优化。", "", "| 空间 | 模型 | 测试准确率均值 | 测试准确率标准差 |", "|---|---|---:|---:|"]
    for space in protocol["spaces"]:
        for name in protocol["models"]:
            group = [row for row in rows if row["space"] == space and row["name"] == name]
            values = [row["test"]["accuracy"] for row in group]
            lines.append(f"| {space} | {name} | {np.mean(values):.1%} | {np.std(values):.1%} |")
    lines += ["", "判定目标：在双曲数据上，标准Poincare距离应恢复真实标签；在欧氏数据上，欧氏距离应恢复真实标签。这个合成实验只验证几何实现和可辨识性，不替代真实冻结向量的泛化测试。"]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
