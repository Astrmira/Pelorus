import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "processed" / "longmemeval_pilot" / "blind_annotation_pack.json"
OUTPUT = ROOT / "data" / "processed" / "longmemeval_pilot" / "annotation_review.json"
STOPWORDS = {
    "about", "after", "again", "been", "before", "between", "could", "from",
    "have", "into", "just", "know", "more", "that", "their", "there", "these",
    "they", "this", "those", "through", "what", "when", "where", "which", "with",
    "would", "your", "you're", "were", "some", "than", "then", "them", "only",
}


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def turn_blocks(text: str) -> list[str]:
    positions = list(re.finditer(r"(?m)^(?:user|assistant): ", text))
    if not positions:
        return [text]
    return [
        text[match.start() : positions[index + 1].start() if index + 1 < len(positions) else len(text)]
        for index, match in enumerate(positions)
    ]


def excerpt(block: str, matched_terms: list[str], radius: int = 450) -> str:
    lowered = block.lower()
    positions = [lowered.find(term) for term in matched_terms if lowered.find(term) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - radius)
    end = min(len(block), center + radius)
    return block[start:end]


def main() -> None:
    items = json.loads(SOURCE.read_text(encoding="utf-8"))
    review = []
    for item in items:
        needles = tokens(f"{item['question']} {item['reference_answer']}")
        candidates = []
        for candidate in item["candidates"]:
            scored = []
            for index, block in enumerate(turn_blocks(candidate["text"])):
                present = needles & tokens(block)
                score = len(present)
                if score:
                    scored.append((score, index, sorted(present), block))
            scored.sort(key=lambda row: (-row[0], row[1]))
            candidates.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "date": candidate["date"],
                    "matched_terms": sorted(set().union(*(set(row[2]) for row in scored[:5]))) if scored else [],
                    "review_blocks": [excerpt(row[3], row[2]) for row in scored[:5]],
                }
            )
        review.append({key: item[key] for key in item if key != "candidates"} | {"candidates": candidates})
    OUTPUT.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成 {len(review)} 题的阅读定位包：{OUTPUT}")


if __name__ == "__main__":
    main()
