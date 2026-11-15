"""仅在旧训练/验证集开发标准几何读出，不重复评估已打开的两个测试集。"""

import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_temporal_readout as base
from geometry_readouts import GeometryReadout
from run_geometry_controlled import save, sha


OUTPUT = base.old.ROOT / 'artifacts/experiments/temporal_geometry_development_v1'
MODELS = ('shared_cosine', 'shared_euclidean', 'bilinear', 'nonlinear_euclidean',
          'poincare_01', 'poincare_1', 'poincare_4', 'poincare_learned')
STYLES = ('seen', 'paraphrase', 'ordinary')
SEEDS = (11, 23, 47)
LRS = (.0005, .005)
STEPS = 600


def assemble(cases, corpus, raw=False):
    if not cases or any(c['split'] not in ('train', 'validation') for c in cases):
        raise ValueError('本开发实验只允许训练与验证链')
    if len({c['split'] for c in cases}) != 1:
        raise ValueError('训练与验证必须分别组装')
    q, d, target = [], [], []
    prefix = 'raw_' if raw else ''
    for case in cases:
        for style in STYLES:
            entries = [x for x in case['queries'] if x['style'] == style]
            if len(entries) != 2 or {x['target_index'] for x in entries} != {0, 1}:
                raise ValueError('每条链每种查询必须恰有两问且目标相反')
            for entry in entries:
                q.append(corpus[prefix+'queries'][entry['query_index']])
                d.append(corpus[prefix+'documents'][case['document_indices']])
                target.append(entry['target_index'])
    return torch.stack(q).double(), torch.stack(d).double(), torch.tensor(target)


def metrics(model, dataset, scale=1.):
    q, d, target = dataset
    with torch.no_grad():
        scores = model(q, d) / scale
        if not torch.isfinite(scores).all():
            raise ValueError('读出出现非有限评分')
        margins = scores.gather(1, target[:, None]).squeeze(1) - scores.gather(1, (1-target)[:, None]).squeeze(1)
        correct = (margins > 1e-6).reshape(-1, 3, 2)
        losses = F.cross_entropy(scores, target, reduction='none').reshape(-1, 3, 2)
        return {style: {'single': float(correct[:, k].double().mean()),
                        'paired': float(correct[:, k].all(-1).double().mean()),
                        'loss': float(losses[:, k].mean())} for k, style in enumerate(STYLES)}


def key(m):
    return ((m['seen']['paired'] + m['paraphrase']['paired']) / 2,
            m['ordinary']['single'], -(m['seen']['loss'] + m['paraphrase']['loss']) / 2)


def fit(name, seed, lr, train, validation, fallback):
    torch.manual_seed(seed)
    model = GeometryReadout(name, train[0].shape[-1], output_dim=8).double()
    q, d, target = train
    with torch.no_grad():
        scores = model(q, d)
        scale = float((scores[:, 0]-scores[:, 1]).square().mean().sqrt().clamp_min(.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best, state, chosen_step, history = fallback, None, None, []
    initial = metrics(model, validation, scale)
    if initial['ordinary']['single'] >= fallback['ordinary']['single'] and key(initial) > key(best):
        best, state, chosen_step = initial, copy.deepcopy(model.state_dict()), 0
    for step in range(1, STEPS + 1):
        ix = rng.choice(len(q), size=32, replace=False)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(q[ix], d[ix]) / scale, target[ix])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0:
            m = metrics(model, validation, scale)
            history.append({'step': step, 'validation': m})
            if m['ordinary']['single'] >= fallback['ordinary']['single'] and key(m) > key(best):
                best, state, chosen_step = m, copy.deepcopy(model.state_dict()), step
    filename = f'{name}_{seed}_{lr}.pt'
    torch.save({'selected': state, 'last': model.state_dict(), 'score_scale': scale}, OUTPUT / filename)
    return {'name': name, 'seed': seed, 'lr': lr, 'chosen_step': chosen_step,
            'fallback': state is None, 'validation': best, 'final_train': metrics(model, train, scale),
            'parameters': sum(p.numel() for p in model.parameters()), 'history': history,
            'checkpoint': filename, 'checkpoint_sha256': sha(OUTPUT / filename)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'protocol.json').exists():
        raise RuntimeError('开发协议已冻结，不允许覆盖；修改需另开版本')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases, corpus, mean, std = base.load_dataset()
    train_cases, val_cases = ([c for c in cases if c['split'] == split] for split in ('train', 'validation'))
    train, validation = assemble(train_cases, corpus), assemble(val_cases, corpus)
    protocol = {'models': MODELS, 'seeds': SEEDS, 'learning_rates': LRS, 'steps': STEPS,
                'batch_size': 32, 'output_dimensions': 8, 'styles': STYLES, 'precision': 'float64',
                'selection': '每模型每种子选验证指标最佳学习率和检查点；普通事实不低于验证选定Cosine基线。先比较两种时间模板成对均值，再普通单问，再时间负损失。',
                'optimization': '统一600步，不采用验证早停；全部曲率共享相同线性映射、半径约束和训练初始无标签分差RMS校准',
                'split': {'train': [c['id'] for c in train_cases], 'validation': [c['id'] for c in val_cases]},
                'status': '探索性开发，只报告6条旧验证链；不是独立泛化或曲率获益证明',
                'frozen_inputs': {str(p.relative_to(base.old.ROOT)): sha(p) for p in
                                  [Path(__file__), Path(__file__).with_name('geometry_readouts.py'),
                                   Path(__file__).with_name('run_temporal_readout.py'), base.DATA/'manifest.json',
                                   base.EMBEDDINGS/'documents.npy', base.EMBEDDINGS/'queries.npy']}}
    save(OUTPUT / 'protocol.json', protocol)
    np.save(OUTPUT/'whitening_mean.npy', mean)
    np.save(OUTPUT/'whitening_std.npy', std)
    cosine = GeometryReadout('raw_cosine', train[0].shape[-1]).double()
    baseline_metrics = {'cosine_whitened': metrics(cosine, validation),
                        'cosine_raw': metrics(cosine, assemble(val_cases, corpus, raw=True))}
    fallback_name = max(baseline_metrics, key=lambda name: key(baseline_metrics[name]))
    rows, registry = [], []
    for name in MODELS:
        for seed in SEEDS:
            candidates = [fit(name, seed, lr, train, validation, baseline_metrics[fallback_name]) for lr in LRS]
            rows.extend(candidates)
            chosen = max(candidates, key=lambda row: key(row['validation']))
            registry.append({k: v for k, v in chosen.items() if k != 'history'})
            save(OUTPUT/'runs.json', rows)
            print(f"{name}，种子{seed}：验证时间成对均值{key(chosen['validation'])[0]:.1%}，回退{chosen['fallback']}", flush=True)
    save(OUTPUT/'selection_registry.json', registry)
    save(OUTPUT/'results.json', {'protocol': protocol, 'baseline_metrics': baseline_metrics,
                               'fallback_name': fallback_name, 'runs': rows,
                               'selection_registry_sha256': sha(OUTPUT/'selection_registry.json')})
    lines = ['# 时间读出标准几何开发检验', '', '仅22条训练链和6条旧验证链；两个已打开的历史测试集均不重新评分。下表是用于选模的验证成绩，不能当独立泛化证据。', '',
             '| 模型 | 参数 | 验证seen成对 | 验证paraphrase成对 | 验证普通单问 | 回退次数 |', '|---|---:|---:|---:|---:|---:|']
    for name in MODELS:
        group = [r for r in registry if r['name'] == name]
        m = lambda s, field: np.mean([r['validation'][s][field] for r in group])
        lines.append(f"| {name} | {group[0]['parameters']} | {m('seen','paired'):.1%} | {m('paraphrase','paired'):.1%} | {m('ordinary','single'):.1%} | {sum(r['fallback'] for r in group)}/3 |")
    lines += ['', f'验证回退基线：{fallback_name}；普通事实门槛{baseline_metrics[fallback_name]["ordinary"]["single"]:.1%}。', '',
              '三种子、两档学习率和600更新步均在运行前冻结，按验证集选中后保存检查点。共享线性欧氏与固定曲率模型参数量相同；双线性和非线性欧氏是更大容量对照。', '',
              '曲率扫描同时改变距离尺度和几何，初始评分校准仅控制其中部分因素；学习曲率与映射尺度混淆，不能从非零曲率推断真实向量流形有非零曲率。', '',
              '下一阶段需要新来源或通过来源/会话/证据去重的新测试集；继续复用旧8条终检将失去盲测意义。']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('标准几何开发实验完成，未评估旧测试集。', flush=True)


if __name__ == '__main__':
    main()
