import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "data" / "processed" / "longmemeval_pilot"
BLIND_PACK = PILOT_DIR / "blind_annotation_pack.json"
OFFICIAL_KEY = PILOT_DIR / "official_evidence_key.json"
OUTPUT = PILOT_DIR / "pilot_annotations.json"

# 仅列出非零候选；其他候选均由人工阅读后判定为无关。
LABELS = {
    "2698e78f": {
        "c05": (3, ["direct", "update"], "explicit", "明确说明现在每周见 Dr. Smith。"),
        "c07": (-1, ["stale"], "explicit", "旧记录说明每两周一次，会误导当前频率判断。"),
    },
    "6a1eabeb": {
        "c02": (3, ["direct", "update"], "explicit", "明确给出更新后的个人最佳成绩 25:50。"),
        "c08": (-1, ["stale"], "explicit", "给出较早的个人最佳成绩 27:12，已被后续成绩更新。"),
    },
    "157a136e": {
        "c02": (2, ["bridge"], "jointly_supported", "明确给出用户年龄 32 岁，需要与祖母年龄联合计算。"),
        "c06": (2, ["bridge"], "jointly_supported", "明确给出祖母年龄 75 岁，需要与用户年龄联合计算。"),
    },
    "2e6d26dc": {
        "c03": (2, ["bridge"], "jointly_supported", "记录 David 的孩子 Jasper，是总数计算的一部分。"),
        "c04": (2, ["bridge"], "jointly_supported", "记录 Charlotte 及 Jasper 等新生儿信息，是总数计算的一部分。"),
        "c05": (2, ["bridge"], "jointly_supported", "记录 Rachel 的孩子 Max，是总数计算的一部分。"),
        "c08": (2, ["bridge"], "jointly_supported", "记录双胞胎 Ava 和 Lily，是总数计算的一部分。"),
    },
    "cc539528": {
        "c03": (3, ["direct"], "explicit", "前后端开发对话中直接给出应学习的后端语言。"),
    },
    "65240037": {
        "c05": (3, ["direct"], "explicit", "直接说明茶树油与载体油应按 1:10 稀释。"),
    },
    "a89d7624": {
        "c06": (3, ["direct", "background"], "explicit", "明确记录 Denver 音乐体验、Brandon Flowers 及偏好。"),
    },
    "32260d93": {
        "c05": (3, ["direct", "background"], "explicit", "明确记录对 Netflix 叙事型单口喜剧专场的偏好。"),
    },
    "76d63226": {
        "c07": (3, ["direct"], "explicit", "直接说明新 Samsung 电视为 55 英寸。"),
    },
    "0862e8bf": {
        "c04": (3, ["direct"], "explicit", "直接说明猫的名字是 Luna。"),
    },
    "6e984301": {
        "c06": (2, ["bridge"], "jointly_supported", "包含雕塑课程开始时间；同一 Session 也含近似干扰表述，需结合时间点读取。"),
        "c08": (2, ["bridge"], "jointly_supported", "记录购买雕塑工具的时间，与课程开始记录联合计算。"),
    },
    "b29f3365": {
        "c01": (3, ["direct", "bridge"], "explicit", "同一 Session 给出已上课六周且音箱购于两周前，可推出四周。"),
        "c02": (1, ["background"], "explicit", "重复确认音箱购于两周前，但单独不能确定当时已上课多久。"),
    },
    "19b5f2b3_abs": {
        "c03": (2, ["abstention_support"], "explicit", "明确记录在日本旅行两周，而非韩国，支持拒绝错误地点前提。"),
        "c04": (-1, ["stale"], "explicit", "只讨论计划去首尔且没有停留时长，若当作已完成韩国旅行会造成误导。"),
    },
    "eeda8a6d_abs": {
        "c05": (2, ["abstention_support"], "explicit", "明确讨论的是 20 加仑鱼缸，不是问题中的 30 加仑鱼缸。"),
        "c06": (2, ["abstention_support"], "explicit", "明确记录 20 加仑新缸和 10 加仑旧缸，支持 30 加仑前提缺失。"),
    },
}


def main() -> None:
    blind_items = json.loads(BLIND_PACK.read_text(encoding="utf-8"))
    official_keys = {
        item["question_id"]: item
        for item in json.loads(OFFICIAL_KEY.read_text(encoding="utf-8"))
    }
    output = {
        "schema_version": "0.1-pilot",
        "annotation_provenance": {
            "annotator": "单一 AI 研究助理",
            "procedure": "依据 Query、参考答案、候选 Session 原文与时间独立判定；未使用任何 embedding 或 metric 分数。",
            "limitation": "这是规范冻结前的 Pilot 标注，不可替代多人独立盲标或全库穷举 qrels。",
            "official_key_usage": "完成语义判断后用于证据覆盖审计，不用于机械生成 relevance。",
        },
        "queries": [],
    }

    for item in blind_items:
        question_id = item["question_id"]
        key = official_keys[question_id]
        annotations = []
        for candidate in item["candidates"]:
            candidate_id = candidate["candidate_id"]
            label = LABELS.get(question_id, {}).get(candidate_id)
            if label is None:
                relevance, roles, support, rationale = (
                    0,
                    ["none"],
                    "unsupported",
                    "候选原文不能帮助回答该问题。",
                )
            else:
                relevance, roles, support, rationale = label
            session_id = key["candidate_mapping"][candidate_id]
            annotations.append(
                {
                    "candidate_id": candidate_id,
                    "session_id": session_id,
                    "relevance": relevance,
                    "roles": roles,
                    "support_strength": support,
                    "rationale_zh": rationale,
                    "is_official_evidence_session": session_id
                    in key["official_answer_session_ids"],
                }
            )
        output["queries"].append(
            {
                "question_id": question_id,
                "question_type": item["question_type"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "is_abstention": item["is_abstention"],
                "annotations": annotations,
            }
        )

    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter(
        annotation["relevance"]
        for query in output["queries"]
        for annotation in query["annotations"]
    )
    print(f"已写入 {len(output['queries'])} 题、{sum(counts.values())} 个候选标注：{dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
