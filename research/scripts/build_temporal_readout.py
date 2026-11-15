"""冻结经原文核验的时间对照；标签不依赖向量或几何模型。"""

import json
import hashlib
from collections import defaultdict
from pathlib import Path

import yaml

import run_low_capacity_experiment as old
from prepare_temporal_review import OUTPUT, SOURCE
from prepare_low_capacity_training_data import INSTRUCTION
from prepare_longmemeval_encoding_dataset import pair_turns, pair_text
from audit_readout_splits import content_fingerprint

ROOT = old.ROOT

# 每条为来源题、主题组、无答案的属性描述、旧值、新值；完整用户原文均已逐项阅读。
ANNOTATIONS = [
    ("6a1eabeb", "running", "my personal-best time in a charity 5K run", "27:12", "25:50"),
    ("6aeb4375", "restaurants", "the number of Korean restaurants I had tried in my city", "three", "four"),
    ("945e3d21", "wellbeing", "how often I attended yoga classes", "twice a week", "three times a week"),
    ("d7c942c3", "shopping_lists", "the grocery-list method my mother was using", "paper list", "same grocery list app"),
    ("71315a70", "art_progress", "the hours I had spent on my abstract ocean sculpture", "5-6 hours", "10-12 hours"),
    ("89941a93", "bicycles", "how many bikes I owned", "three bikes", "four bikes"),
    ("ce6d2d27", "cocktails", "the weekday of my cocktail-making class", "Thursday", "Fridays"),
    ("07741c44", "shoe_storage", "where I was keeping my old sneakers", "under my bed", "shoe rack in my closet"),
    ("a1eacc2a", "writing", "the number of short stories I had completed since writing regularly", "four", "7"),
    ("184da446", "reading", "my current page in A Short History of Nearly Everything", "200", "220"),
    ("031748ae", "engineering", "how many engineers I led as a Senior Software Engineer", "4 engineers", "five engineers"),
    ("4d6b87c8", "film_viewing", "the number of titles on my to-watch list", "20", "25"),
    ("41698283", "photography", "the new camera lens I mentioned using", "50mm prime", "70-200mm zoom"),
    ("2698e78f", "wellbeing", "how often I saw my therapist Dr. Smith", "every two weeks", "every week"),
    ("b6019101", "film_viewing", "the number of MCU films I had watched in the preceding three months", "4 MCU films", "5 MCU films"),
    ("45dc21b6", "cooking", "the number of Emma's recipes I had tried", "two", "3"),
    ("6071bd76", "coffee", "ounces of water per tablespoon of coffee in my French press", "6 ounces", "5 ounces"),
    ("e493bb7c", "art_location", "where my Ethereal Dreams painting was displayed", "living room sofa", "above my bed"),
    ("01493427", "collecting", "the number of postcards added since I resumed collecting", "17", "25"),
    ("18bc8abd", "cooking", "my favorite BBQ sauce", "Sweet Baby Ray's", "Kansas City Masterpiece"),
    ("1cea1afa", "social_media", "my Instagram follower count", "500", "600"),
    ("ed4ddc30", "food_stock", "how many dozen eggs were stocked in my refrigerator", "30 dozen", "20 dozen"),
    ("b01defab", "reading", "whether I had finished The Nightingale by Kristin Hannah", "not yet finished", "finished"),
    ("0e4e4c46", "boardgames", "my highest score in Ticket to Ride", "124 points", "132 points"),
    ("4b24c848", "clothes", "the number of tops I had bought from H&M", "three tops", "five tops"),
    ("603deb26", "cocktails", "how many times I had made a Negroni since Emma taught me", "5 times", "10 times"),
    ("59524333", "gym_schedule", "the time of day I usually went to the gym", "7:00 pm", "6:00 pm"),
    ("affe2881", "birdwatching", "the number of bird species I had seen in my local park", "27", "32"),
    ("50635ada", "travel_rewards", "my United Airlines frequent-flyer status", "Premier Silver", "Premier Gold"),
    ("f685340e", "tennis", "how often I played tennis with friends at the local park", "weekly", "every other week"),
    ("cc5ded98", "coding", "my daily time spent on coding exercises", "an hour", "two hours"),
    ("7401057b", "travel_rewards", "the number of free Hilton nights my points could redeem", "single free night", "two free nights"),
    ("c7dc5443", "volleyball", "my recreational volleyball league record", "3-2", "5-2"),
    ("06db6396", "art_progress", "the number of projects completed since starting painting classes", "4 projects", "5th project"),
    ("3ba21379", "model_building", "the vehicle model I was working on", "Ford Mustang Shelby GT350R", "Ford F-150 pickup truck"),
    ("dad224aa", "morning_routine", "my usual Saturday wake-up time", "8:30 am", "7:30 am"),
]
EXCLUSIONS = {
    "618f13b2": "同一天六次记录早于四次记录，计数与记录时间冲突。",
    "c6853660": "后一条只是考虑把咖啡限额改为两杯，不是已经改变。",
    "2133c1b5": "记录相隔约六个月，居住时长只由一月增为三月，时间一致性存疑。",
    "7a87bd0c": "记录相隔数月，持续时长仅增加一周，不适合作为严格时间链。",
    "89941a94": "与89941a93复用同一事实文本，并存在gravel/hybrid表述冲突。",
    "07741c45": "与07741c44复用同一事实链。",
    "a2f3aa27": "后一数值只是接近1300的估计，不是明确更新值。",
    "5a4f22c0": "早期标记用户句没有明确旧公司，缺少独立可核验的旧状态。",
    "22d2cb42": "只有一个明确状态，不能形成两状态对照。",
    "eace081b": "只有一个明确计划位置，不能形成两状态对照。",
}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build():
    reviewed = {row["id"]: row for row in json.loads((OUTPUT / "review.json").read_text(encoding="utf-8"))}
    source = {str(row["question_id"]): row for row in json.loads(SOURCE.read_text(encoding="utf-8"))}
    used_content = set()
    for split in ("train", "validation"):
        for row in old.read_jsonl(old.DATA_DIR / f"{split}_documents.jsonl"):
            used_content.add(sha(" ".join(row["text"].splitlines()[1:]).strip()))
    for row in old.read_jsonl(old.TEST_DATA_DIR / "documents.jsonl"):
        used_content.add(sha(" ".join(row["text"].splitlines()[1:]).strip()))
    cases, documents, queries = [], [], []
    for qid, group, subject, early_value, late_value in ANNOTATIONS:
        row = reviewed[qid]
        marked = [(session, pair) for session in row["sessions"] for pair in session["pairs"] if pair["has_answer"]]
        if len(marked) != 2:
            raise ValueError(f"预期恰好两个经核验的证据Pair：{qid}")
        marked.sort(key=lambda item: item[0]["date"])
        state_docs, evidence = [], []
        for state, (session, pair), answer in zip(("earlier", "later"), marked, (early_value, late_value), strict=True):
            turns = source[qid]["haystack_sessions"][session["session_index"]]
            paired = pair_turns(turns)[pair["pair_index"]]
            users = [turn for turn in paired if turn["role"] == "user"]
            if len(users) != 1:
                raise ValueError("证据必须包含一条明确用户陈述")
            text = pair_text(session["date"], users)
            # ID不含新旧状态，模型与并列处理不能从ID中获得答案。
            doc_id = "td_" + sha(qid + "\0" + text)[:24]
            documents.append({"id": doc_id, "text": text})
            state_docs.append(doc_id)
            evidence.append({"document_id": doc_id, "date": session["date"], "source_session_index": session["session_index"],
                             "source_pair_index": pair["pair_index"], "answer": answer, "state": state,
                             "session_fingerprint": content_fingerprint(turns), "user_fingerprint": content_fingerprint(users),
                             "previously_encoded_pair": sha(" ".join(pair["text"].splitlines()[1:]).strip()) in used_content,
                             "support_text": text})
        if evidence[0]["date"] >= evidence[1]["date"]:
            raise ValueError("事实链记录时间必须严格递增")
        case = {"id": qid, "group": group, "subject": subject, "previously_used_source": row["previously_used"],
                "evidence": evidence, "candidate_ids": sorted(state_docs), "queries": [],
                "annotation": "单一AI阅读并核验两个用户陈述；标签回答各记录所陈述的状态，不推断现实状态的持续区间。"}
        for style in ("seen", "paraphrase", "ordinary"):
            for index, (state, value) in enumerate(zip(("earlier", "later"), (early_value, late_value), strict=True)):
                if style == "seen":
                    text = f"According to the {state} of the two dated updates, what did I report about {subject}?"
                elif style == "paraphrase":
                    cue = "first entry chronologically" if index == 0 else "most recent entry chronologically"
                    text = f"Find the evidence about {subject} in the {cue}."
                else:
                    text = f"Find the update where I reported {value} regarding {subject}."
                query_id = "tq_" + sha(qid + style + state)[:24]
                queries.append({"id": query_id, "text": INSTRUCTION + text})
                case["queries"].append({"query_id": query_id, "style": style, "state": state, "target": state_docs[index], "text": text})
        neutral_id = "tq_" + sha(qid + "neutral")[:24]
        queries.append({"id": neutral_id, "text": INSTRUCTION + f"What did I report about {subject}?"})
        case["neutral_query_id"] = neutral_id
        cases.append(case)
    # 同主题或共享证据必须位于同一切分；本次不让旧实验用过的主题进入新任务留出集。
    groups = defaultdict(list)
    for case in cases:
        groups[case["group"]].append(case)
    fresh_groups = [name for name, items in groups.items() if all(not case["previously_used_source"] and
                    not any(item["previously_encoded_pair"] for item in case["evidence"]) for case in items)]
    fresh_groups.sort(key=lambda name: sha("temporal-20260913:" + name))
    holdout_groups, count = [], 0
    for group in fresh_groups:
        if count >= 8:
            break
        holdout_groups.append(group)
        count += len(groups[group])
    if count < 8:
        raise ValueError("未使用的事实链不足以建立8条留出链")
    remaining = sorted((name for name in groups if name not in holdout_groups), key=lambda name: sha("validation-20260913:" + name))
    validation_groups, count = [], 0
    for group in remaining:
        if count >= 6:
            break
        validation_groups.append(group)
        count += len(groups[group])
    for case in cases:
        case["split"] = "holdout" if case["group"] in holdout_groups else "validation" if case["group"] in validation_groups else "train"
    owners = defaultdict(set)
    for case in cases:
        for item in case["evidence"]:
            for key in ("session_fingerprint", "user_fingerprint"):
                owners[(key, item[key])].add(case["split"])
    if any(len(value) > 1 for value in owners.values()):
        raise ValueError("主题分组后仍发现跨切分共享证据，需先合并事实链")
    write_rows(OUTPUT / "documents.jsonl", documents)
    write_rows(OUTPUT / "queries.jsonl", queries)
    save = lambda name, value: (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    save("annotations.json", cases)
    save("exclusions.json", EXCLUSIONS)
    manifest = {"source_sha256": old.sha256_file(SOURCE), "annotation_script_sha256": old.sha256_file(Path(__file__)),
                "chains": len(cases), "documents": len(documents), "queries": len(queries),
                "splits": {name: [case["id"] for case in cases if case["split"] == name] for name in ("train", "validation", "holdout")},
                "labels": "在给定两条记录中检索相应时间顺序的证据；每个时间查询恰好一个目标。不是全Haystack检索标签。",
                "limitations": ["原文来自LongMemEval构建的对话基准，不能称为真实用户自然纵向观测。",
                                "查询由AI构造，标签仅单一AI核验，尚无第二标注者。",
                                "只测两候选条件读出；时间元数据规则可直接解决该微型任务。",
                                "语料标注者看过留出文本，但建模前未查看新任务留出指标；历史使用标记仅核验所选Pair及来源题。"],
                "input_sha256": {name: old.sha256_file(OUTPUT / name) for name in ("documents.jsonl", "queries.jsonl", "annotations.json", "exclusions.json")}}
    save("manifest.json", manifest)
    embeddings = ROOT / "artifacts" / "embeddings" / "temporal_readout"
    config = yaml.safe_load((ROOT / "configs" / "low_capacity_train_encoding.yml").read_text(encoding="utf-8"))
    config["experiment"] = {"seed": 20260913, "dataset_manifest": str((OUTPUT / "manifest.json").relative_to(ROOT)), "output_dir": str(embeddings.relative_to(ROOT))}
    config["inputs"] = {name: str((OUTPUT / f"{name}.jsonl").relative_to(ROOT)) for name in ("documents", "queries")}
    config["embedding"]["batch_size"] = 8
    config["retry"]["max_attempts"] = 3
    config["retry"]["backoff_max_seconds"] = 30
    config["cache"] = {"document_cache": str((embeddings / "cache_documents.jsonl").relative_to(ROOT)),
                       "query_cache": str((embeddings / "cache_queries.jsonl").relative_to(ROOT)),
                       "document_vectors": str((embeddings / "documents.npy").relative_to(ROOT)),
                       "query_vectors": str((embeddings / "queries.npy").relative_to(ROOT)),
                       "document_ids": str((embeddings / "document_ids.json").relative_to(ROOT)),
                       "query_ids": str((embeddings / "query_ids.json").relative_to(ROOT)),
                       "runtime_manifest": str((embeddings / "runtime_manifest.json").relative_to(ROOT))}
    (ROOT / "configs" / "temporal_readout_encoding.yml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps({"事实链": len(cases), "文档": len(documents), "查询": len(queries),
                      "切分": {key: len(value) for key, value in manifest["splits"].items()},
                      "估算字符数": sum(len(row["text"]) for row in documents + queries)}, ensure_ascii=False))


if __name__ == "__main__":
    build()
