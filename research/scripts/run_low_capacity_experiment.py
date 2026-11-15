import copy
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_cosine_baseline import evaluate_official, evaluate_pilot, group_qrels, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "longmemeval_low_capacity"
TEST_DATA_DIR = ROOT / "data" / "processed" / "longmemeval_turn_pairs"
EMBEDDING_DIR = ROOT / "artifacts" / "embeddings" / "longmemeval_low_capacity"
TEST_EMBEDDING_DIR = ROOT / "artifacts" / "embeddings" / "longmemeval_turn_pairs"
OUTPUT_DIR = ROOT / "artifacts" / "experiments" / "low_capacity_readout_v3_bounded_residual"
SEEDS = (11, 23, 47)
OUTPUT_DIM = 32
EPOCHS = 60
PATIENCE = 8
BATCH_QUERIES = 8
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.1
RADIUS_MIN = 0.05
RADIUS_BUDGET = 2.0
CURVATURE = 1.0
EPS = 1e-8


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=EPS)


def lorentz_distance(q: torch.Tensor, d: torch.Tensor, curvature: float = CURVATURE) -> torch.Tensor:
    sqrt_c = math.sqrt(curvature)
    q_radius = torch.linalg.norm(q, dim=-1)
    d_radius = torch.linalg.norm(d, dim=-1)
    cosine = torch.sum(normalize(q) * normalize(d), dim=-1).clamp(-1.0, 1.0)
    alpha = (
        torch.cosh(sqrt_c * q_radius) * torch.cosh(sqrt_c * d_radius)
        - torch.sinh(sqrt_c * q_radius) * torch.sinh(sqrt_c * d_radius) * cosine
    )
    return torch.acosh(alpha.clamp_min(1.0 + 1e-7)) / sqrt_c


class ResidualScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_logit = nn.Parameter(torch.tensor(-4.0))

    def base_score(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        return normalize(documents) @ normalize(query)

    def combine(
        self, query: torch.Tensor, documents: torch.Tensor, geometric_score: torch.Tensor
    ) -> torch.Tensor:
        residual_weight = torch.sigmoid(self.residual_logit)
        bounded_geometric_score = torch.tanh(geometric_score)
        return (
            self.base_score(query, documents) + residual_weight * bounded_geometric_score
        ) / TEMPERATURE


class SharedMahalanobis(ResidualScorer):
    def __init__(self, input_dim: int, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

    def score(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        query_projected = self.projection(query)
        document_projected = self.projection(documents)
        geometric_score = -torch.sum((query_projected - document_projected) ** 2, dim=-1)
        return self.combine(query, documents, geometric_score)


class FixedWhitenedCosine(nn.Module):
    def score(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        return normalize(documents) @ normalize(query) / TEMPERATURE


class EuclideanTwoTower(ResidualScorer):
    def __init__(self, input_dim: int, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.query_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.document_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.query_radius = nn.Linear(input_dim, 1)
        self.document_radius = nn.Linear(input_dim, 1)

    def point(self, x: torch.Tensor, projection: nn.Linear, radius: nn.Linear) -> torch.Tensor:
        direction = normalize(projection(x))
        bounded_radius = RADIUS_MIN + (RADIUS_BUDGET - RADIUS_MIN) * torch.sigmoid(radius(x))
        return direction * bounded_radius

    def score(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        query_point = self.point(query, self.query_projection, self.query_radius)
        document_points = self.point(documents, self.document_projection, self.document_radius)
        geometric_score = -torch.sum((query_point - document_points) ** 2, dim=-1)
        return self.combine(query, documents, geometric_score)


class HyperbolicTwoTower(ResidualScorer):
    def __init__(self, input_dim: int, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.query_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.document_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.query_radius = nn.Linear(input_dim, 1)
        self.document_radius = nn.Linear(input_dim, 1)

    def tangent(self, x: torch.Tensor, projection: nn.Linear, radius: nn.Linear) -> torch.Tensor:
        direction = normalize(projection(x))
        bounded_radius = RADIUS_MIN + (RADIUS_BUDGET - RADIUS_MIN) * torch.sigmoid(radius(x))
        return direction * bounded_radius

    def score(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        query_tangent = self.tangent(query, self.query_projection, self.query_radius)
        document_tangent = self.tangent(documents, self.document_projection, self.document_radius)
        geometric_score = -lorentz_distance(
            query_tangent.expand_as(document_tangent), document_tangent
        )
        return self.combine(query, documents, geometric_score)


class ProductTwoTower(ResidualScorer):
    def __init__(self, input_dim: int, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.query_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.document_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.query_radius = nn.Linear(input_dim, 1)
        self.document_radius = nn.Linear(input_dim, 1)
        self.split = output_dim // 2

    def components(
        self, x: torch.Tensor, projection: nn.Linear, radius: nn.Linear
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = projection(x)
        euclidean = normalize(projected[..., : self.split])
        hyperbolic_direction = normalize(projected[..., self.split :])
        hyperbolic_radius = RADIUS_MIN + (RADIUS_BUDGET - RADIUS_MIN) * torch.sigmoid(radius(x))
        return euclidean, hyperbolic_direction * hyperbolic_radius

    def score(self, query: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        query_euclidean, query_hyperbolic = self.components(
            query, self.query_projection, self.query_radius
        )
        document_euclidean, document_hyperbolic = self.components(
            documents, self.document_projection, self.document_radius
        )
        euclidean_distance = torch.sum(
            (query_euclidean.expand_as(document_euclidean) - document_euclidean) ** 2, dim=-1
        )
        hyperbolic_distance = lorentz_distance(
            query_hyperbolic.expand_as(document_hyperbolic), document_hyperbolic
        )
        geometric_score = -(0.5 * euclidean_distance + 0.5 * hyperbolic_distance**2)
        return self.combine(query, documents, geometric_score)


MODEL_CLASSES = {
    "B0_fixed_whitened_cosine": FixedWhitenedCosine,
    "B1_shared_mahalanobis": SharedMahalanobis,
    "B2_euclidean_two_tower": EuclideanTwoTower,
    "H1_hyperbolic_two_tower": HyperbolicTwoTower,
    "H2_euclidean_hyperbolic_product": ProductTwoTower,
}


def fit_whitening(documents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = documents.mean(axis=0)
    std = np.maximum(documents.std(axis=0), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_whitening(vectors: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((vectors - mean) / std).astype(np.float32)


def load_split(split: str, mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    embedding_root = EMBEDDING_DIR / split
    document_ids = json.loads((embedding_root / "document_ids.json").read_text(encoding="utf-8"))
    query_ids = json.loads((embedding_root / "query_ids.json").read_text(encoding="utf-8"))
    documents = apply_whitening(np.load(embedding_root / "documents.npy"), mean, std)
    queries = apply_whitening(np.load(embedding_root / "queries.npy"), mean, std)
    metadata = read_jsonl(DATA_DIR / f"{split}_metadata.jsonl")
    document_index = {document_id: index for index, document_id in enumerate(document_ids)}
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    examples = []
    for row in metadata:
        candidate_indices = [document_index[item] for item in row["candidate_ids"]]
        positive_ids = set(row["positive_ids"])
        positive_mask = np.asarray([item in positive_ids for item in row["candidate_ids"]])
        examples.append(
            {
                "query_id": row["query_id"],
                "query_index": query_index[row["query_id"]],
                "candidate_indices": candidate_indices,
                "positive_mask": positive_mask,
            }
        )
    return {"documents": documents, "queries": queries, "examples": examples}


def multi_positive_loss(scores: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[positive_mask], dim=0)


def evaluate_validation(model: nn.Module, split: dict[str, Any], device: torch.device) -> float:
    model.eval()
    reciprocal_ranks = []
    with torch.no_grad():
        documents = torch.from_numpy(split["documents"]).to(device)
        queries = torch.from_numpy(split["queries"]).to(device)
        for example in split["examples"]:
            scores = model.score(
                queries[example["query_index"]], documents[example["candidate_indices"]]
            )
            order = torch.argsort(scores, descending=True).cpu().numpy()
            positive_mask = example["positive_mask"]
            first = next(rank for rank, position in enumerate(order, 1) if positive_mask[position])
            reciprocal_ranks.append(1.0 / first)
    return float(np.mean(reciprocal_ranks))


def train_model(
    model: nn.Module,
    train: dict[str, Any],
    validation: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model.to(device)
    if parameter_count(model) == 0:
        validation_mrr = evaluate_validation(model, validation, device)
        return model, {
            "best_validation_mrr": validation_mrr,
            "best_epoch": 0,
            "history": [],
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_documents = torch.from_numpy(train["documents"]).to(device)
    train_queries = torch.from_numpy(train["queries"]).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_mrr = evaluate_validation(model, validation, device)
    best_epoch = 0
    stale_epochs = 0
    history = [{"epoch": 0, "loss": None, "validation_mrr": best_mrr}]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = np.random.permutation(len(train["examples"]))
        losses = []
        optimizer.zero_grad()
        for batch_start in range(0, len(order), BATCH_QUERIES):
            batch_loss = torch.zeros((), device=device)
            batch_indices = order[batch_start : batch_start + BATCH_QUERIES]
            for example_index in batch_indices:
                example = train["examples"][int(example_index)]
                scores = model.score(
                    train_queries[example["query_index"]],
                    train_documents[example["candidate_indices"]],
                )
                positive_mask = torch.from_numpy(example["positive_mask"]).to(device)
                batch_loss = batch_loss + multi_positive_loss(scores, positive_mask)
            batch_loss = batch_loss / len(batch_indices)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(batch_loss.detach().cpu()))

        validation_mrr = evaluate_validation(model, validation, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_mrr": validation_mrr})
        if validation_mrr > best_mrr + 1e-8:
            best_mrr = validation_mrr
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, {"best_validation_mrr": best_mrr, "best_epoch": best_epoch, "history": history}


def rank_test(
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict[str, list[tuple[str, float]]]:
    documents_meta = read_jsonl(TEST_DATA_DIR / "documents.jsonl")
    queries_meta = read_jsonl(TEST_DATA_DIR / "queries.jsonl")
    document_ids = json.loads((TEST_EMBEDDING_DIR / "document_ids.json").read_text(encoding="utf-8"))
    query_ids = json.loads((TEST_EMBEDDING_DIR / "query_ids_instructed.json").read_text(encoding="utf-8"))
    documents = torch.from_numpy(
        apply_whitening(np.load(TEST_EMBEDDING_DIR / "documents.npy"), mean, std)
    ).to(device)
    queries = torch.from_numpy(
        apply_whitening(np.load(TEST_EMBEDDING_DIR / "queries_instructed.npy"), mean, std)
    ).to(device)
    document_index = {document_id: index for index, document_id in enumerate(document_ids)}
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    candidates_by_query: dict[str, list[str]] = defaultdict(list)
    for row in documents_meta:
        candidates_by_query[str(row["question_id"])].append(str(row["document_id"]))

    rankings = {}
    model.eval()
    with torch.no_grad():
        for query in queries_meta:
            query_id = str(query["query_id"])
            candidates = candidates_by_query[query_id]
            indices = [document_index[item] for item in candidates]
            scores = model.score(queries[query_index[query_id]], documents[indices])
            order = torch.argsort(scores, descending=True).cpu().numpy()
            rankings[query_id] = [
                (candidates[int(position)], float(scores[int(position)].cpu())) for position in order
            ]
    return rankings


def evaluate_test(rankings: dict[str, list[tuple[str, float]]]) -> dict[str, Any]:
    official = group_qrels(read_jsonl(TEST_DATA_DIR / "official_qrels.jsonl"))
    pilot = group_qrels(read_jsonl(TEST_DATA_DIR / "pilot_qrels.jsonl"))
    queries = read_jsonl(TEST_DATA_DIR / "queries.jsonl")
    query_metadata = {str(row["query_id"]): row for row in queries}
    official_summary, _ = evaluate_official(rankings, official)
    pilot_summary, _ = evaluate_pilot(rankings, pilot, query_metadata)
    return {"official": official_summary, "pilot": pilot_summary}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required = [
        EMBEDDING_DIR / split / name
        for split in ("train", "validation")
        for name in ("documents.npy", "queries.npy", "document_ids.json", "query_ids.json")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少编码产物：" + ", ".join(missing))

    train_documents_raw = np.load(EMBEDDING_DIR / "train" / "documents.npy")
    mean, std = fit_whitening(train_documents_raw)
    train = load_split("train", mean, std)
    validation = load_split("validation", mean, std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "whitening_mean.npy", mean)
    np.save(OUTPUT_DIR / "whitening_std.npy", std)

    results: dict[str, Any] = {
        "protocol": {
            "train_queries": len(train["examples"]),
            "validation_queries": len(validation["examples"]),
            "test_queries": 14,
            "test_isolation": "Pilot Test 的 question_id 未参与训练、Whitening 或 early stopping。",
            "training_truth": "仅使用官方 has_answer 派生的多正例；无等级和误导监督。",
            "selection": "每个模型和 Seed 按 Validation MRR early stopping，Test 不参与选择。",
            "seeds": list(SEEDS),
            "output_dim": OUTPUT_DIM,
            "device": str(device),
        },
        "models": {},
        "input_sha256": {
            "training_manifest": sha256_file(DATA_DIR / "manifest.json"),
            "test_document_vectors": sha256_file(TEST_EMBEDDING_DIR / "documents.npy"),
            "test_query_vectors": sha256_file(TEST_EMBEDDING_DIR / "queries_instructed.npy"),
        },
    }

    input_dim = train["documents"].shape[1]
    for model_name, model_class in MODEL_CLASSES.items():
        runs = []
        model_seeds = (SEEDS[0],) if model_class is FixedWhitenedCosine else SEEDS
        for seed in model_seeds:
            torch.manual_seed(seed)
            model = model_class() if model_class is FixedWhitenedCosine else model_class(input_dim)
            model, training = train_model(model, train, validation, seed, device)
            rankings = rank_test(model, mean, std, device)
            metrics = evaluate_test(rankings)
            checkpoint = OUTPUT_DIR / f"{model_name}_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            runs.append(
                {
                    "seed": seed,
                    "parameters": parameter_count(model),
                    "training": training,
                    "test": metrics,
                    "checkpoint": checkpoint.name,
                }
            )
            print(
                f"{model_name} seed={seed} val_mrr={training['best_validation_mrr']:.4f} "
                f"test_mrr={metrics['official']['mrr']:.4f} test_ndcg10={metrics['pilot']['ndcg@10']:.4f}"
            )
        results["models"][model_name] = {
            "parameters": runs[0]["parameters"],
            "runs": runs,
            "aggregate": {
                "official_mrr_mean": float(np.mean([run["test"]["official"]["mrr"] for run in runs])),
                "official_mrr_std": float(np.std([run["test"]["official"]["mrr"] for run in runs])),
                "official_recall10_mean": float(
                    np.mean([run["test"]["official"]["recall@10"] for run in runs])
                ),
                "pilot_ndcg10_mean": float(np.mean([run["test"]["pilot"]["ndcg@10"] for run in runs])),
                "pilot_ndcg10_std": float(np.std([run["test"]["pilot"]["ndcg@10"] for run in runs])),
                "pilot_positive_recall10_mean": float(
                    np.mean([run["test"]["pilot"]["positive_recall@10"] for run in runs])
                ),
                "pilot_misleading_hit10_mean": float(
                    np.mean([run["test"]["pilot"]["misleading_hit@10"] for run in runs])
                ),
            },
        }

    output = OUTPUT_DIR / "results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: row["aggregate"] for name, row in results["models"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
