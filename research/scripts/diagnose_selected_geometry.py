"""检查开发选中的双曲模型是否实际使用了非平坦距离，不做额外选模。"""

import json
import torch

import run_temporal_readout as base
from run_temporal_geometry_development import OUTPUT, assemble
from run_geometry_controlled import save, sha
from geometry_readouts import GeometryReadout, tangent_distance_squared


def main():
    torch.set_num_threads(1)
    cases, corpus, _, _ = base.load_dataset()
    dataset = assemble([c for c in cases if c['split'] == 'validation'], corpus)
    registry = json.loads((OUTPUT/'selection_registry.json').read_text(encoding='utf-8'))
    rows = []
    for r in registry:
        if not r['name'].startswith('poincare') or r['fallback']:
            continue
        path = OUTPUT/r['checkpoint']
        if sha(path) != r['checkpoint_sha256']:
            raise ValueError('冻结检查点哈希不匹配')
        model = GeometryReadout(r['name'], 1536, 8).double()
        model.load_state_dict(torch.load(path, weights_only=True)['selected'])
        with torch.no_grad():
            q, d, _ = dataset
            qv, dv = model.coordinates(q)[:, None], model.coordinates(d)
            flat = (qv-dv).square().sum(-1)
            curved = tangent_distance_squared(qv, dv, model.curvature())
            ratios = (curved / flat.clamp_min(1e-12)).sqrt()
            rows.append({'name': r['name'], 'seed': r['seed'], 'c': float(model.curvature()),
                         'mean_tangent_radius': float(torch.cat((qv.flatten(0,1),dv.flatten(0,1))).norm(dim=-1).mean()),
                         'distance_ratio_mean': float(ratios.mean()), 'distance_ratio_max': float(ratios.max()),
                         'curvature_ablation_prediction_change': float((curved.argmin(-1) != flat.argmin(-1)).double().mean()),
                         'interpretation': '只在6条旧验证链上将已训练坐标的距离改为欧氏，未重训，不视为新的独立实验'})
    save(OUTPUT/'curvature_diagnostic.json', rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
