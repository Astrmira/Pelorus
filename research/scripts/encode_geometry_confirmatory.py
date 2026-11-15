"""先用公开锚点验证后端稳定性，再编码冻结的新测试输入。"""

import json
from pathlib import Path
import numpy as np
import yaml

import encode_embeddings_api as api
from prepare_geometry_confirmatory import ROOT, DATA, EXP, EMBED
from run_geometry_controlled import save, sha


def verify_inputs():
    protocol = json.loads((EXP/'protocol.json').read_text(encoding='utf-8'))
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    if sha(DATA/'manifest.json') != protocol['input_manifest_sha256']:
        raise ValueError('确认性输入清单发生变化')
    for name, expected in manifest['input_sha256'].items():
        if sha(DATA/name) != expected:
            raise ValueError('确认性输入文件发生变化：'+name)
    return protocol, manifest


def check_anchors(protocol, manifest, config):
    rows = []
    total_usage = {'prompt_tokens':0,'total_tokens':0,'requests':0}
    for kind in ('documents','queries'):
        all_rows = api.read_jsonl(DATA/(kind+'.jsonl'))
        targets = [x for x in manifest['anchors'] if x['kind']==kind]
        ids = {a['id'] for a in targets}
        selected = [r for r in all_rows if r['id'] in ids]
        cache_key = 'document_cache' if kind=='documents' else 'query_cache'
        cache, usage, models = api.encode_rows(kind+'公开锚点',selected,ROOT/config['cache'][cache_key],config,
                                              api.EMBEDDING_API_BASE_URL.rstrip('/')+'/embeddings', api.EMBEDDING_API_KEY)
        if models and models != {'Qwen3-Embedding-8B'}:
            raise ValueError('公开锚点返回了不同模型')
        for key in usage:
            total_usage[key] += usage[key]
        old_embed = ROOT/'artifacts/embeddings/temporal_readout'
        old_vectors = np.load(old_embed/(kind+'.npy'))
        old_ids = json.loads((old_embed/('document_ids.json' if kind=='documents' else 'query_ids.json')).read_text(encoding='utf-8'))
        index = {v:i for i,v in enumerate(old_ids)}
        for a in targets:
            before = old_vectors[index[a['old_id']]].astype(np.float64)
            after = np.asarray(cache[a['id']]['embedding'],dtype=np.float64)
            before /= np.linalg.norm(before)
            after /= np.linalg.norm(after)
            cosine = float(before @ after)
            error = float(np.linalg.norm(before-after))
            rows.append({**a,'cosine':cosine,'relative_l2':error})
    gate = protocol['backend_anchor_gate']
    passed = all(r['cosine'] >= gate['minimum_cosine'] and r['relative_l2'] <= gate['maximum_relative_l2'] for r in rows)
    report = {'passed':passed,'gate':gate,'anchors':rows,'usage_this_run':total_usage,
              'api_origin':'ir.cofi.date','limitations':'8个公开锚点只能检测明显漂移，不证明全后端权重完全一致。'}
    save(EXP/'backend_anchor_audit.json',report)
    if not passed:
        raise RuntimeError('公开锚点未通过预定一致性阈值，已停止新测试编码')
    print(f"公开锚点一致性通过，最小Cosine={min(r['cosine'] for r in rows):.9f}。",flush=True)


def main():
    protocol, manifest = verify_inputs()
    config_path = ROOT/'configs/geometry_confirmatory_encoding.yml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    lock_path = EXP/'evaluation_lock.json'
    evaluator = ROOT/'scripts/evaluate_geometry_confirmatory.py'
    locks = {'protocol_sha256':sha(EXP/'protocol.json'),'evaluator_sha256':sha(evaluator),
             'encoder_sha256':sha(Path(__file__)),'config_sha256':sha(config_path)}
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding='utf-8')) != locks:
            raise ValueError('编码前冻结的评估脚本或协议发生变化')
    else:
        save(lock_path,locks)
    check_anchors(protocol,manifest,config)
    import sys
    sys.argv = [str(ROOT/'scripts/encode_embeddings_api.py'),'--config',str(config_path)]
    api.main()
    save(EXP/'encoded_files_manifest.json', {name:sha(EMBED/name) for name in
                                           ('documents.npy','queries.npy','document_ids.json','query_ids.json','runtime_manifest.json')})


if __name__ == '__main__':
    main()
