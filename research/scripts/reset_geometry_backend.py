"""保留旧锚点失败记录，在当前同一服务端重编码训练/验证输入。"""

import json
from pathlib import Path
import sys
import yaml

import encode_embeddings_api as api
import run_temporal_readout as base
import run_temporal_geometry_development as development
from run_geometry_controlled import save, sha
from prepare_geometry_confirmatory import ROOT, DATA, EXP, EMBED


CURRENT_DATA = ROOT/'data/processed/temporal_geometry_current_backend_v1'
CURRENT_EMBED = ROOT/'artifacts/embeddings/temporal_geometry_current_backend_v1'
CURRENT_DEV = ROOT/'artifacts/experiments/temporal_geometry_current_backend_v1'
CURRENT_EXP = ROOT/'artifacts/experiments/geometry_confirmatory_v2'
CURRENT_TEST_EMBED = ROOT/'artifacts/embeddings/geometry_confirmatory_v2'


def configure(path, data, embed, template):
    cfg = yaml.safe_load(template.read_text(encoding='utf-8'))
    cfg['experiment'] = {'seed':20261016,'dataset_manifest':str((data/'manifest.json').relative_to(ROOT)), 'output_dir':str(embed.relative_to(ROOT))}
    cfg['inputs'] = {kind:str((data/(kind+'.jsonl')).relative_to(ROOT)) for kind in ('documents','queries')}
    cfg['embedding']['batch_size'] = 32
    for key,value in cfg['cache'].items():
        cfg['cache'][key] = str((embed/Path(value).name).relative_to(ROOT))
    path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')


def prepare():
    CURRENT_DATA.mkdir(parents=True,exist_ok=True)
    old_data = base.DATA
    cases = [c for c in json.loads((old_data/'annotations.json').read_text(encoding='utf-8')) if c['split'] in ('train','validation')]
    wanted_docs = {d for c in cases for d in c['candidate_ids']}
    wanted_queries = {q['query_id'] for c in cases for q in c['queries']} | {c['neutral_query_id'] for c in cases}
    save(CURRENT_DATA/'annotations.json',cases)
    for name,ids in (('documents',wanted_docs),('queries',wanted_queries)):
        rows = [r for r in api.read_jsonl(old_data/(name+'.jsonl')) if r['id'] in ids]
        (CURRENT_DATA/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    manifest = {'chains':len(cases),'documents':len(wanted_docs),'queries':len(wanted_queries),
                'purpose':'相同训练/验证文本由当前API重新编码；只改变向量后端快照，超参数和标签不变。',
                'input_sha256':{name:sha(CURRENT_DATA/name) for name in ('documents.jsonl','queries.jsonl','annotations.json')}}
    save(CURRENT_DATA/'manifest.json',manifest)
    save(EXP/'stopped.json',{'reason':'1/8公开锚点超过预定归一化L2阈值，未开始新测试评分。',
                           'audit_sha256':sha(EXP/'backend_anchor_audit.json'),
                           'continuation':'geometry_confirmatory_v2：相同新测试数据，全部训练/验证文本重新编码，再选择并冻结。'})
    template = ROOT/'configs/geometry_confirmatory_encoding.yml'
    configure(ROOT/'configs/temporal_geometry_current_backend_encoding.yml',CURRENT_DATA,CURRENT_EMBED,template)
    configure(ROOT/'configs/geometry_confirmatory_v2_encoding.yml',DATA,CURRENT_TEST_EMBED,template)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


def main():
    if (CURRENT_DEV/'protocol.json').exists():
        raise RuntimeError('当前后端开发选择已冻结，禁止重跑')
    prepare()
    sys.argv = [str(Path(api.__file__)),'--config',str(ROOT/'configs/temporal_geometry_current_backend_encoding.yml')]
    api.main()
    # 原训练实现支持从相同格式的数据路径加载；保持其超参数与模型定义一致。
    base.DATA, base.EMBEDDINGS = CURRENT_DATA, CURRENT_EMBED
    development.OUTPUT = CURRENT_DEV
    development.main()
    CURRENT_EXP.mkdir(parents=True,exist_ok=True)
    protocol = json.loads((EXP/'protocol.json').read_text(encoding='utf-8'))
    registry = json.loads((CURRENT_DEV/'selection_registry.json').read_text(encoding='utf-8'))
    protocol.update(status='所有训练/验证文本由当前API重新编码并选模；编码新测试文本前冻结',
                    development_registry_sha256=sha(CURRENT_DEV/'selection_registry.json'),
                    checkpoints_sha256={r['checkpoint']:sha(CURRENT_DEV/r['checkpoint']) for r in registry},
                    whitening_sha256={name:sha(CURRENT_DEV/name) for name in ('whitening_mean.npy','whitening_std.npy')},
                    backend_reset={'prior_stopped_protocol_sha256':sha(EXP/'protocol.json'),
                                   'anchor_audit_sha256':sha(EXP/'backend_anchor_audit.json'),
                                   'current_training_manifest_sha256':sha(CURRENT_EMBED/'runtime_manifest.json'),
                                   'reason':'保持原阈值失败记录，不混用旧后端训练向量；全部重编码再训练。'})
    protocol['backend_anchor_gate'] = {'policy':'旧后端阈值已失败并保留；v2训练/验证/终检均使用当前API独立缓存。'}
    protocol['code_sha256'].update({name:sha(ROOT/'scripts'/name) for name in
                                   ('reset_geometry_backend.py','evaluate_geometry_confirmatory.py','evaluate_geometry_confirmatory_v2.py')})
    save(CURRENT_EXP/'protocol.json',protocol)
    print('当前后端重新训练和选择已冻结，尚未评估新测试集。',flush=True)


if __name__ == '__main__':
    main()
