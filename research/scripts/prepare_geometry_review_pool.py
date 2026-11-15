"""从未使用来源中准备下一轮阅读池；不自动赋予时间关系标签。"""

import json
from collections import Counter
from pathlib import Path

from audit_readout_splits import content_fingerprint
from prepare_longmemeval_encoding_dataset import pair_turns, pair_text, document_id
from run_geometry_controlled import save, sha


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
OUTPUT = DATA / 'geometry_review_pool'


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def main():
    source_path = ROOT / 'data/raw/longmemeval_s_cleaned.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    used_ids, wanted_docs = set(), set()
    for split in ('train', 'validation'):
        used_ids.update(r['query_id'] for r in jsonl(DATA/'longmemeval_low_capacity'/f'{split}_metadata.jsonl'))
    used_ids.update(r['query_id'] for r in jsonl(DATA/'longmemeval_turn_pairs/queries.jsonl'))
    # 所有历史实际候选都纳入隔离，不限于正例。
    for folder in ('longmemeval_low_capacity', 'longmemeval_turn_pairs'):
        for path in (DATA/folder).glob('*documents.jsonl'):
            wanted_docs.update(r.get('id', r.get('document_id')) for r in jsonl(path))
    temporal = []
    for folder in ('temporal_readout', 'temporal_fresh_test'):
        temporal += json.loads((DATA/folder/'annotations.json').read_text(encoding='utf-8'))
    used_ids.update(r['id'] for r in temporal)
    known_positions = {(r['id'], e['source_session_index'], e['source_pair_index'])
                       for r in temporal for e in r['evidence']}
    known_sessions, known_users = set(), set()
    for row in source:
        qid = str(row['question_id'])
        if qid not in used_ids:
            continue
        for si, (sid, turns) in enumerate(zip(row['haystack_session_ids'], row['haystack_sessions'], strict=True)):
            for pi, pair in enumerate(pair_turns(turns)):
                if document_id(qid, str(sid), pi) in wanted_docs or (qid, si, pi) in known_positions:
                    known_sessions.add(content_fingerprint(turns))
                    users = [t for t in pair if t['role'] == 'user']
                    if users:
                        known_users.add(content_fingerprint(users))
    exclusions = json.loads((DATA/'temporal_readout/exclusions.json').read_text(encoding='utf-8'))
    pool, excluded = [], []
    for row in source:
        qid = str(row['question_id'])
        if row['question_type'] not in ('knowledge-update', 'temporal-reasoning', 'multi-session'):
            continue
        reason = None
        if qid in used_ids:
            reason = '已进入历史训练或评估'
        elif qid.endswith('_abs'):
            reason = '不可回答题不自动转为时间链'
        elif qid in exclusions:
            reason = '继承历史标注排除：'+exclusions[qid]
        if reason:
            excluded.append({'id': qid, 'reason': reason})
            continue
        evidence = []
        for si, (sid, date, turns) in enumerate(zip(row['haystack_session_ids'], row['haystack_dates'], row['haystack_sessions'], strict=True)):
            for pi, pair in enumerate(pair_turns(turns)):
                if not any(t.get('has_answer') for t in pair):
                    continue
                users = [t for t in pair if t['role'] == 'user']
                evidence.append({'session_index': si, 'pair_index': pi, 'date': date,
                                 'session_fingerprint': content_fingerprint(turns),
                                 'user_fingerprint': content_fingerprint(users) if users else None,
                                 'user_count': len(users), 'text': pair_text(str(date), pair)})
        if len(evidence) < 2 or len({e['session_index'] for e in evidence}) < 2:
            reason = '不足两条跨会话标记证据'
        elif any(e['session_fingerprint'] in known_sessions or e['user_fingerprint'] in known_users for e in evidence):
            reason = '与历史候选共享会话或用户原文'
        if reason:
            excluded.append({'id': qid, 'reason': reason})
        else:
            pool.append({'id': qid, 'question_type': row['question_type'], 'original_question': row['question'],
                         'original_answer': row['answer'], 'evidence': evidence, 'status': '待语义核验，未生成任务标签'})
    # 阅读池内部共享证据形成连通组，未来不得跨训练/验证/测试拆分。
    parents = {r['id']: r['id'] for r in pool}
    def root(qid):
        while parents[qid] != qid:
            qid = parents[qid]
        return qid
    owners = {}
    for row in pool:
        for e in row['evidence']:
            for field in ('session_fingerprint', 'user_fingerprint'):
                value = e[field]
                if value is None:
                    continue
                key = (field, value)
                if key in owners:
                    parents[root(row['id'])] = root(owners[key])
                owners[key] = row['id']
    for row in pool:
        row['evidence_group'] = root(row['id'])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save(OUTPUT/'review.json', pool)
    save(OUTPUT/'exclusions.json', excluded)
    summary = {'source_sha256': sha(source_path), 'script_sha256': sha(Path(__file__)),
               'candidate_questions': len(pool), 'evidence_groups': len({r['evidence_group'] for r in pool}),
               'types': dict(Counter(r['question_type'] for r in pool)),
               'exclusion_reasons': dict(Counter(r['reason'] for r in excluded)),
               'limits': ['仅精确会话/用户原文去重，主题或改写重复仍需审阅',
                          '多会话/时间推理标记不保证有可用状态链；需要逐项核验',
                          '本阅读池不是测试集，不能把候选数量写成已核验事实链数量']}
    save(OUTPUT/'manifest.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
