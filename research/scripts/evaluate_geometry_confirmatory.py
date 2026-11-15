"""一次性评估冻结的新链；主比较与所有反事实/消融按协议分别报告。"""

import json
from pathlib import Path
import numpy as np
import torch

from geometry_readouts import GeometryReadout
from prepare_geometry_confirmatory import ROOT, DATA, EXP, EMBED, DEV
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


def cluster_interval(values, topics, replicates=20000, seed=20261016):
    """保留簇内相关性；每次重采样主题簇后以原题数加权。"""
    values = np.asarray(values,dtype=np.float64)
    topics = np.asarray(topics)
    groups = sorted(set(topics))
    sums = np.array([values[topics==g].sum() for g in groups])
    counts = np.array([(topics==g).sum() for g in groups])
    rng = np.random.default_rng(seed)
    ix = rng.integers(0,len(groups),(replicates,len(groups)))
    sample_means = sums[ix].sum(1)/counts[ix].sum(1)
    return {'estimate':float(values.mean()),'ci95':np.quantile(sample_means,[.025,.975]).tolist(),
            'chains':len(values),'clusters':len(groups)}


def load_vectors(manifest):
    runtime = json.loads((EMBED/'runtime_manifest.json').read_text(encoding='utf-8'))
    if runtime['request_model'] != 'Qwen3-Embedding-8B' or runtime['dimensions'] != 1536:
        raise ValueError('新向量的模型或维度不符合协议')
    if runtime['response_models'] != ['Qwen3-Embedding-8B'] or runtime['api_origin'] != 'ir.cofi.date':
        raise ValueError('新向量返回模型或来源不符合协议')
    result = {}
    mean,std = (np.load(DEV/name) for name in ('whitening_mean.npy','whitening_std.npy'))
    for kind in ('documents','queries'):
        if runtime['input_sha256'][kind] != manifest['input_sha256'][kind+'.jsonl']:
            raise ValueError('新向量输入哈希不符')
        ids_file = 'document_ids.json' if kind=='documents' else 'query_ids.json'
        ids = json.loads((EMBED/ids_file).read_text(encoding='utf-8'))
        vectors = np.load(EMBED/(kind+'.npy'))
        if vectors.shape != (len(ids),1536) or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors,axis=1)==0):
            raise ValueError('新向量非有限、维度不符或含零向量')
        result['raw_'+kind] = torch.from_numpy(vectors/np.linalg.norm(vectors,axis=1,keepdims=True)).double()
        # 精确保留开发阶段float32 Whitening/归一化后再转float64的顺序。
        whitened = ((vectors-mean)/std).astype(np.float32)
        result[kind] = torch.from_numpy(whitened/np.linalg.norm(whitened,axis=1,keepdims=True)).double()
        result[kind+'_index'] = {value:i for i,value in enumerate(ids)}
    return result


def pair_summary(margins, queries, style):
    rows = [(m,q) for m,q in zip(margins,queries,strict=True) if q['style']==style]
    if not rows:
        return None
    correct = [m>1e-6 for m,q in rows]
    families = sorted({q['family'] for m,q in rows})
    paired = [all(m>1e-6 for m,q in rows if q['family']==f) for f in families]
    return {'single':float(np.mean(correct)),'paired':float(np.mean(paired)),
            'family_paired':[int(v) for v in paired]}


def score_case(model, case, corpus, scale=1.0, raw=False):
    prefix = 'raw_' if raw else ''
    queries = case['queries']
    q = corpus[prefix+'queries'][[corpus['queries_index'][r['id']] for r in queries]]
    scores, margins = {}, {}
    with torch.no_grad():
        for condition in ('dated','swapped_dates','no_date'):
            docs = corpus[prefix+'documents'][[corpus['documents_index'][i] for i in case['document_variants'][condition]]]
            s = model(q,docs.unsqueeze(0).expand(len(q),-1,-1))/scale
            if not torch.isfinite(s).all():
                raise ValueError('终检读出产生非有限评分')
            targets = torch.tensor([1-r['target_index'] if condition=='swapped_dates' and r['style']=='record' else r['target_index'] for r in queries])
            margin = s.gather(1,targets[:,None]).squeeze(1)-s.gather(1,(1-targets)[:,None]).squeeze(1)
            scores[condition] = s
            margins[condition] = margin.tolist()
        out = {'id':case['id'],'topic':case['topic'],'evidence_group':case['evidence_group'],
               'dated_record':pair_summary(margins['dated'],queries,'record'),
               'swapped_record':pair_summary(margins['swapped_dates'],queries,'record'),
               'dated_ordinary':pair_summary(margins['dated'],queries,'ordinary')}
        record_families = sorted({r['family'] for r in queries if r['style']=='record'})
        follows = []
        for f in record_families:
            indices = [i for i,r in enumerate(queries) if r['style']=='record' and r['family']==f]
            follows.append(all(margins[c][i]>1e-6 for c in ('dated','swapped_dates') for i in indices))
        out['strict_date_following'] = float(np.mean(follows))
        neutral = next(i for i,r in enumerate(queries) if r['style']=='neutral')
        gap = float(scores['dated'][neutral,0]-scores['dated'][neutral,1])
        out['neutral_paired'] = float(gap>1e-6 and -gap>1e-6)
        if out['neutral_paired'] != 0:
            raise ValueError('无条件同问不应同时支持两个相反目标')
        out['record_predictions'] = scores['dated'].argmax(-1)[:6].tolist()
        if 'event_review' in case:
            out['events'] = {c:pair_summary(margins[c],queries,'event') for c in scores}
    return out


def main():
    if (EXP/'results.json').exists():
        raise RuntimeError('独立终检已经完成，禁止覆盖或重复选模')
    protocol, manifest = verify_inputs()
    lock = json.loads((EXP/'evaluation_lock.json').read_text(encoding='utf-8'))
    if lock['protocol_sha256'] != sha(EXP/'protocol.json') or lock['evaluator_sha256'] != sha(Path(__file__)):
        raise ValueError('编码前锁定的协议或评估代码不一致')
    if 'backend_reset' in protocol:
        current = ROOT/'artifacts/embeddings/temporal_geometry_current_backend_v1/runtime_manifest.json'
        if sha(current) != protocol['backend_reset']['current_training_manifest_sha256']:
            raise ValueError('当前后端重新编码的训练快照发生变化')
    elif not json.loads((EXP/'backend_anchor_audit.json').read_text(encoding='utf-8'))['passed']:
        raise ValueError('公开锚点没有通过')
    for name,expected in protocol['code_sha256'].items():
        if sha(ROOT/'scripts'/name) != expected:
            raise ValueError('冻结模型实现或标注代码发生变化')
    if protocol['development_registry_sha256'] != sha(DEV/'selection_registry.json'):
        raise ValueError('开发选择记录发生变化')
    for name,expected in {**protocol['checkpoints_sha256'],**protocol['whitening_sha256']}.items():
        if sha(DEV/name) != expected:
            raise ValueError('冻结模型或Whitening文件发生变化')
    vector_hashes = json.loads((EXP/'encoded_files_manifest.json').read_text(encoding='utf-8'))
    if any(sha(EMBED/name) != h for name,h in vector_hashes.items()):
        raise ValueError('编码产物哈希发生变化')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus = load_vectors(manifest)
    cases = json.loads((DATA/'annotations.json').read_text(encoding='utf-8'))
    # 编码/评分前的二次语义审查：purchase可能指预订或收到，不能共用事件先后标签。
    # 该题的主要记录顺序与普通事实目标仍然明确；只排除其探索性事件子任务。
    secondary_event_exclusions = {'gpt4_2312f94c':'事件查询使用purchase，正文区分预订与收到；事件标签语义不唯一。'}
    for case in cases:
        if case['id'] in secondary_event_exclusions:
            case.pop('event_review',None)
    event_count = sum('event_review' in c for c in cases)
    registry = json.loads((DEV/'selection_registry.json').read_text(encoding='utf-8'))
    baseline_name = json.loads((DEV/'results.json').read_text(encoding='utf-8'))['fallback_name']
    rows, ablations = [], []
    for name in ('cosine_raw','cosine_whitened'):
        model = GeometryReadout('raw_cosine',1536).double()
        evaluated = [score_case(model,c,corpus,raw=name=='cosine_raw') for c in cases]
        rows.append({'name':name,'seed':None,'cases':evaluated})
    for entry in registry:
        if entry['fallback']:
            model = GeometryReadout('raw_cosine',1536).double()
            scale,raw = 1.0,baseline_name=='cosine_raw'
        else:
            model = GeometryReadout(entry['name'],1536,8).double()
            state = torch.load(DEV/entry['checkpoint'],weights_only=True,map_location='cpu')
            model.load_state_dict(state['selected'])
            scale,raw = state['score_scale'],False
        evaluated = [score_case(model,c,corpus,scale,raw) for c in cases]
        rows.append({'name':entry['name'],'seed':entry['seed'],'checkpoint':entry['checkpoint'],'cases':evaluated})
        if entry['name'].startswith('poincare') and not entry['fallback']:
            original_name = model.name
            model.name = 'shared_euclidean'
            flat = [score_case(model,c,corpus,scale,raw) for c in cases]
            ablations.append({'name':original_name,'seed':entry['seed'],'cases':flat,
                              'record_prediction_change':float(np.mean([a!=b for original,replaced in zip(evaluated,flat,strict=True)
                                                                       for a,b in zip(original['record_predictions'],replaced['record_predictions'],strict=True)]))})
        print(f"已评估冻结模型：{entry['name']}，种子{entry['seed']}。",flush=True)
    def model_chain(name,field,table=rows):
        selected = [r for r in table if r['name']==name]
        return np.mean([[c[field]['paired'] if field!='dated_ordinary' else c[field]['single'] for c in r['cases']] for r in selected],axis=0)
    topics = [c['topic'] for c in cases]
    p = model_chain('poincare_1','dated_record')
    e = model_chain('shared_euclidean','dated_record')
    main_result = cluster_interval(p-e,topics)
    ordinary = cluster_interval(model_chain('poincare_1','dated_ordinary')-model_chain('shared_euclidean','dated_ordinary'),topics)
    seed_deltas = []
    for seed in protocol['seeds']:
        means = {r['name']:np.mean([c['dated_record']['paired'] for c in r['cases']]) for r in rows if r['seed']==seed}
        seed_deltas.append(means['poincare_1']-means['shared_euclidean'])
    passed = main_result['ci95'][0]>0 and main_result['estimate']>=.05 and min(seed_deltas)>0 and ordinary['ci95'][0]>-.05
    summary = []
    for name in ('cosine_raw','cosine_whitened',*protocol['models']):
        selected = [r for r in rows if r['name']==name]
        row = {'name':name,'dated_record':float(model_chain(name,'dated_record').mean()),
               'dated_ordinary':float(model_chain(name,'dated_ordinary').mean()),
               'swapped_record':float(model_chain(name,'swapped_record').mean()),
               'strict_date_following':float(np.mean([c['strict_date_following'] for r in selected for c in r['cases']])),
               'event_dated':float(np.mean([c['events']['dated']['paired'] for r in selected for c in r['cases'] if 'events' in c])),
               'event_no_date':float(np.mean([c['events']['no_date']['paired'] for r in selected for c in r['cases'] if 'events' in c])),
               'event_swapped_dates':float(np.mean([c['events']['swapped_dates']['paired'] for r in selected for c in r['cases'] if 'events' in c]))}
        summary.append(row)
    curvature_increment = cluster_interval(p-model_chain('poincare_1','dated_record',ablations),topics)
    result = {'protocol_sha256':sha(EXP/'protocol.json'),'evaluation_lock':lock,'vector_hashes':vector_hashes,
              'summary':summary,'primary_comparison':main_result,'ordinary_noninferiority':ordinary,
              'primary_seed_differences':seed_deltas,'predefined_success':passed,
              'curvature_ablation_increment':curvature_increment,'runs':rows,'ablations':ablations,
              'secondary_event_exclusions_before_scoring':secondary_event_exclusions,
              'date_rule':{'record_dated':1.0,'record_swapped_dates':1.0,'meaning':'已验证Date行严格递增且swap目标同步反转；不用于事件正文顺序。'}}
    save(EXP/'results.json',result)
    lines = ['# 50条新证据链的独立几何读出检验', '',
             f"预定主要标准：{'满足' if passed else '未满足'}。50条证据链按26个保守主题簇重采样；每链先平均三措辞和三种子。主要任务回答消息记录顺序，事件顺序另列。", '',
             '| 读出 | 记录顺序成对 | 普通事实单问 | 日期交换成对 | 严格跟随日期反转 |', '|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['name']} | {r['dated_record']:.2%} | {r['dated_ordinary']:.2%} | {r['swapped_record']:.2%} | {r['strict_date_following']:.2%} |")
    lines += ['', f"预定主比较（K=-1双曲减同参数欧氏）：{main_result['estimate']:.2%}；配对主题簇bootstrap 95%区间 [{main_result['ci95'][0]:.2%}, {main_result['ci95'][1]:.2%}]。",
              f"三个种子的差值：{', '.join(f'{v:.2%}' for v in seed_deltas)}。",
              f"普通事实差值：{ordinary['estimate']:.2%}；95%区间 [{ordinary['ci95'][0]:.2%}, {ordinary['ci95'][1]:.2%}]；非劣阈值为-5个百分点。",
              f"保留双曲坐标只换欧氏距离的成对成绩差：{curvature_increment['estimate']:.2%}，95%区间 [{curvature_increment['ci95'][0]:.2%}, {curvature_increment['ci95'][1]:.2%}]。", '',
              f'## 正文可识别的{event_count}条事件顺序子集（探索性）', '',
              '| 读出 | 原日期 | 去消息Date行 | 交换消息日期 |', '|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['name']} | {r['event_dated']:.2%} | {r['event_no_date']:.2%} | {r['event_swapped_dates']:.2%} |")
    lines += ['', '去Date行保留正文的事件日期或相对时间；并未移除所有日期信息。事件子集目标按正文先后，交换消息日期不改事件标签。', '',
              '原始标注包含14个事件候选；编码/评分前二次审查发现gpt4_2312f94c的purchase可能指预订或收到，因此只保留其主要记录任务，事件子任务排除，最终事件子集13条。该排除已写入编码前冻结的评估代码。', '',
              '无条件相同查询的成对正确率全部为0。日期规则的记录任务正确率100%，仅作定义与标签自检。', '',
              '局限：来源为LongMemEval构造对话；单一AI标注；主题与既有训练可能重叠；仅两候选证据选择。簇bootstrap条件于冻结训练集，26簇对小效应功效有限。其余曲率/模型/子集均探索性，没有按终检重新选模。', '',
              '全部输入、模型、预定规则、编码前评估代码与向量产物哈希均已验证。此结果不能外推到一般黎曼度量、其他编码器或全Haystack检索。']
    (EXP/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('新证据链独立检验完成，主要成功标准：'+str(passed),flush=True)


if __name__ == '__main__':
    main()
