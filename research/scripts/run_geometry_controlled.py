"""已知几何的受控恢复实验，区分自然分布与距离分歧压力测试。"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from geometry_readouts import GeometryReadout, tangent_distance_squared


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/experiments/geometry_controlled_v2'
MODELS = ('raw_cosine', 'shared_cosine', 'shared_euclidean', 'bilinear',
          'nonlinear_euclidean', 'poincare_01', 'poincare_1', 'poincare_4', 'poincare_learned')
SEEDS = (11, 23, 47)
LRS = (0.01, 0.03)
STEPS = 600


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def latent_triples(rng, count):
    # 两种标签几何共享相同的切空间采样分布，不按目标距离筛选自然样本。
    angles = rng.uniform(-np.pi, np.pi, (count, 3))
    radii = rng.uniform(0.05, 4.0, (count, 3))
    return np.stack((radii * np.cos(angles), radii * np.sin(angles)), -1)


def label_distances(latent, curvature):
    points = torch.as_tensor(latent, dtype=torch.float64)
    return tangent_distance_squared(points[:, :1], points[:, 1:], curvature).numpy()


def generate(count, seed, projection, pressure=False):
    rng = np.random.default_rng(seed)
    if pressure:
        chunks, total = [], 0
        while total < count:
            block = latent_triples(rng, count)
            flat = label_distances(block, 0).argmin(-1)
            curved = label_distances(block, 1).argmin(-1)
            block = block[flat != curved]
            chunks.append(block)
            total += len(block)
        latent = np.concatenate(chunks)[:count]
    else:
        latent = latent_triples(rng, count)
    # 每组三点独立抽样，候选顺序独立随机；目标索引并不固定。
    swap = rng.integers(0, 2, count).astype(bool)
    latent[swap, 1:] = latent[swap, 1:][:, ::-1]
    return {'latent': latent, 'observed': latent @ projection,
            'euclidean_target': label_distances(latent, 0).argmin(-1),
            'hyperbolic_target': label_distances(latent, 1).argmin(-1)}


def tensors(data, space):
    observed = torch.tensor(data['observed'], dtype=torch.float64)
    return observed[:, 0], observed[:, 1:], torch.tensor(data[space + '_target'])


def evaluate(model, data):
    q, docs, target = data
    with torch.no_grad():
        scores = model(q, docs)
        if not torch.isfinite(scores).all():
            raise ValueError('读出分数出现非有限值')
        margin = scores.gather(1, target[:, None]).squeeze(1) - scores.gather(1, (1-target)[:, None]).squeeze(1)
        correct = margin > 1e-10
        return {'accuracy': float(correct.double().mean()), 'loss': float(F.cross_entropy(scores, target)),
                'correct': correct.int().tolist()}


def selection_key(metrics):
    return metrics['accuracy'], -metrics['loss']


def fit(name, seed, lr, train, validation):
    torch.manual_seed(seed)
    model = GeometryReadout(name, 16).double()
    if name == 'raw_cosine':
        return model, {'step': 0, 'validation': evaluate(model, validation), 'parameters': 0}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    q, docs, target = train
    rng = np.random.default_rng(seed)
    best = evaluate(model, validation)
    selected = copy.deepcopy(model.state_dict())
    selected_step, history = 0, []
    for step in range(1, STEPS + 1):
        index = rng.choice(len(q), size=256, replace=False)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(q[index], docs[index]), target[index])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0:
            m = evaluate(model, validation)
            history.append({'step': step, 'validation_accuracy': m['accuracy'], 'validation_loss': m['loss']})
            if selection_key(m) > selection_key(best):
                best, selected, selected_step = m, copy.deepcopy(model.state_dict()), step
    last_train = evaluate(model, train)
    model.load_state_dict(selected)
    return model, {'step': selected_step, 'validation': best, 'parameters': sum(p.numel() for p in model.parameters()),
                   'last_train_accuracy': last_train['accuracy'], 'history': history,
                   'selected_c': float(model.curvature())}


def paired_interval(a, b):
    # 对相同测试三元组先平均种子，再按三元组重采样；种子不当作独立数据。
    delta = np.mean(a, axis=0) - np.mean(b, axis=0)
    rng = np.random.default_rng(20260919)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(4000)])
    return {'difference': float(delta.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'units': len(delta), 'interpretation': '条件于本训练集的逐三元组配对区间，不包含训练集重抽样不确定性'}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'protocol.json').exists():
        raise RuntimeError('实验目录已冻结；请使用新版本目录，禁止覆盖')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    projection = np.linalg.qr(np.random.default_rng(911).normal(size=(16, 2)))[0].T
    settings = {'train': (2400, 1921, False), 'validation': (800, 1922, False),
                'natural_test': (2000, 1923, False), 'disagreement_test': (1000, 1924, True)}
    protocol = {'models': MODELS, 'seeds': SEEDS, 'lrs': LRS, 'steps': STEPS, 'batch_size': 256,
                'spaces': ['euclidean', 'hyperbolic'], 'settings': settings,
                'latent': '二维切空间半径U(0.05,4)、角度U(-pi,pi)，同一采样仅改变标签度量；正交线性嵌入16维，无丢失信息',
                'curvature': 'K=-c，比较c=0/0.1/1/4/学习；优化是切空间参数的AdamW，不声称测试了黎曼优化器',
                'selection': '每模型每种子用验证准确率、负交叉熵选择学习率和检查点；所有选择冻结后才评估测试',
                'primary': '自然测试上poincare_1减shared_euclidean；两个标签几何分别报告，不合并',
                'secondary': '其余模型及欧氏/双曲排序分歧压力集均作探索性诊断，不挑最高测试模型',
                'controls': '共享线性距离模型32参数，学习曲率多1参数；双线性64参数，非线性更强；精度均float64；共享半径上限6',
                'limitations': ['标签由目标几何构造，匹配的oracle满分是自检，不是实际任务上的几何收益证据',
                               '压力集按距离分歧筛选，不能用其结果代表自然分布',
                               '曲率与坐标尺度存在混淆，学习c不作为恢复真实曲率的证据'],
                'script_sha256': {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('geometry_readouts.py'))}}
    save(OUTPUT / 'protocol.json', protocol)
    datasets = {name: generate(n, seed, projection, pressure) for name, (n, seed, pressure) in settings.items()}
    for name, data in datasets.items():
        np.savez(OUTPUT / (name + '.npz'), **data)
    save(OUTPUT / 'data_manifest.json', {name: sha(OUTPUT / (name + '.npz')) for name in datasets})
    registry, models, candidates = [], {}, []
    for space in protocol['spaces']:
        train, validation = (tensors(datasets[name], space) for name in ('train', 'validation'))
        for name in MODELS:
            for seed in ((11,) if name == 'raw_cosine' else SEEDS):
                best, chosen = None, None
                for lr in ((0.0,) if name == 'raw_cosine' else LRS):
                    model, row = fit(name, seed, lr, train, validation)
                    row.update(space=space, name=name, seed=seed, lr=lr)
                    candidates.append(row)
                    if best is None or selection_key(row['validation']) > selection_key(best['validation']):
                        best, chosen = row, model
                filename = f'{space}_{name}_{seed}.pt'
                torch.save(chosen.state_dict(), OUTPUT / filename)
                registry.append({**best, 'checkpoint': filename, 'checkpoint_sha256': sha(OUTPUT / filename)})
                models[filename] = chosen
                save(OUTPUT / 'candidates.json', candidates)
                print(f"{space}，{name}，种子{seed}：验证{best['validation']['accuracy']:.1%}，学习率{best['lr']}", flush=True)
    save(OUTPUT / 'selection_registry.json', registry)
    registry_sha = sha(OUTPUT / 'selection_registry.json')
    print('全部选择已经冻结，开始独立测试与压力测试。', flush=True)
    rows = []
    for row in registry:
        metrics = {split: evaluate(models[row['checkpoint']], tensors(datasets[split], row['space']))
                   for split in ('natural_test', 'disagreement_test')}
        rows.append({**row, 'evaluation': metrics})
    primary = {}
    for space in protocol['spaces']:
        successes = lambda name: [r['evaluation']['natural_test']['correct'] for r in rows if r['space'] == space and r['name'] == name]
        primary[space] = paired_interval(successes('poincare_1'), successes('shared_euclidean'))
    save(OUTPUT / 'results.json', {'protocol': protocol, 'selection_registry_sha256': registry_sha,
                                  'runs': rows, 'primary_comparison': primary})
    lines = ['# 受控几何恢复实验 v2', '', '保留v1初步结果；本版本独立生成数据并冻结协议。自然分布与按距离分歧筛选的压力集分开报告。所有模型从相同16维冻结输入学习；只有标签生成器使用二维真实坐标。', '',
             '| 标签几何 | 读出 | 参数量 | 自然测试均值 | 压力测试均值 |', '|---|---|---:|---:|---:|']
    for space in protocol['spaces']:
        for name in MODELS:
            group = [r for r in rows if r['space'] == space and r['name'] == name]
            values = [np.mean([r['evaluation'][s]['accuracy'] for r in group]) for s in ('natural_test', 'disagreement_test')]
            lines.append(f'| {space} | {name} | {group[0]["parameters"]} | {values[0]:.2%} | {values[1]:.2%} |')
    lines += ['', '预定主要比较：固定c=1双曲读出减共享线性欧氏读出；置信区间按测试三元组配对，不把种子当独立数据。']
    for space, stat in primary.items():
        lines.append(f"- {space}：差值 {stat['difference']:.2%}，95%区间 [{stat['ci95'][0]:.2%}, {stat['ci95'][1]:.2%}]。")
    lines += ['', '上述区间条件于固定训练集，不覆盖换训练集的不确定性。压力集不用于宣称一般分布优越性；额外模型和曲率比较均属探索性。', '',
              '标准距离参考：https://arxiv.org/abs/1705.08039 。实现以等价asinh形式计算精确球内测地距离；c=0在正交切空间坐标下严格退化为欧氏距离。', '',
              '本轮不是Qwen时间读出实验，也不能证明真实文本向量有曲率优势。']
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('受控几何实验完成。', flush=True)


if __name__ == '__main__':
    main()
