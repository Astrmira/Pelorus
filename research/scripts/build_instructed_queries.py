import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "longmemeval_turn_pairs" / "encoding_queries.jsonl"
OUTPUT = ROOT / "data" / "processed" / "longmemeval_turn_pairs" / "encoding_queries_instructed.jsonl"
MANIFEST = ROOT / "data" / "processed" / "longmemeval_turn_pairs" / "instructed_query_manifest.json"
INSTRUCTION = (
    "Instruct: Given a personal conversation history, retrieve the passages that provide "
    "the evidence needed to answer the question.\nQuery: "
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    rows = read_jsonl(SOURCE)
    output_rows = [
        {"id": str(row["id"]), "text": INSTRUCTION + str(row["text"])} for row in rows
    ]
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    manifest = {
        "variant": "instructed_query_v1",
        "instruction": INSTRUCTION,
        "query_count": len(output_rows),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUTPUT),
        "document_input_changed": False,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
