"""在完全隔离的合成数据上执行训练到评分的集成检查。"""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np


def main():
    production=Path(__file__).resolve().parents[1]
    temp=Path(tempfile.mkdtemp(prefix='pubmed-geometry-smoke-',dir='/hy-tmp'))
    (temp/'scripts').mkdir()
    for name in ('run_pubmed_geometry.py','prepare_pubmed_geometry.py','geometry_readouts.py','pubmed_cloud_pipeline.py'):
        shutil.copy2(production/'scripts'/name,temp/'scripts'/name)
    sys.path.insert(0,str(temp/'scripts'))
    import run_pubmed_geometry as model
    from prepare_pubmed_geometry import DATA,EMBED,EXP,save,sha
    assert DATA.is_relative_to(temp) and EMBED.is_relative_to(temp) and EXP.is_relative_to(temp)
    DATA.mkdir(parents=True);EMBED.mkdir(parents=True);EXP.mkdir(parents=True)
    rng=np.random.default_rng(20261016)
    ids=[f'd{i}' for i in range(100)];qids=[f'q{i}' for i in range(100)]
    rows=[]
    for i in range(100):
        choices=rng.choice(100,10,replace=False).tolist();target=i%10
        rows.append({'id':qids[i],'split':'train' if i<64 else 'validation' if i<80 else 'test',
                     'cluster':f'branch_{i//5}','target_id':ids[choices[target]],'target_index':target,
                     'candidates':[{'id':ids[v],'type':'target' if j==target else 'random'} for j,v in enumerate(choices)]})
    save(DATA/'annotations.json',rows)
    for kind,identifiers in (('documents',ids),('queries',qids)):
        (DATA/(kind+'.jsonl')).write_text(''.join(json.dumps({'id':i,'text':'隔离合成检查'})+'\n' for i in identifiers))
        np.save(EMBED/(kind+'.npy'),rng.normal(size=(100,1536)).astype(np.float32))
        save(EMBED/(('document' if kind=='documents' else 'query')+'_ids.json'),identifiers)
    hashes={n:sha(DATA/n) for n in ('annotations.json','documents.jsonl','queries.jsonl')}
    save(DATA/'manifest.json',{'input_sha256':hashes,'scope':'仅流水线集成自检，不是真实实验'})
    save(EMBED/'runtime_manifest.json',{'request_model':'Qwen3-Embedding-8B','dimensions':1536,
                                      'input_sha256':{k:hashes[k+'.jsonl'] for k in ('documents','queries')}})
    save(EXP/'encoding_lock.json',{'manifest_sha256':sha(DATA/'manifest.json'),
                                 'code_sha256':{n:sha(temp/'scripts'/n) for n in ('run_pubmed_geometry.py','prepare_pubmed_geometry.py','geometry_readouts.py','pubmed_cloud_pipeline.py')}})
    model.SEEDS=(11,);model.LRS=(.0005,);model.STEPS=100
    model.main()
    result=json.loads((EXP/'results.json').read_text())
    assert len(result['summary'])==8 and result['verification']['all_checkpoint_hashes_match']
    assert all(len(r['local'])==20 and len(r['global'])==20 for r in result['runs'])
    record={'passed':True,'scope':'隔离合成数据上的训练、验证选模、保存、交叉评分及全库评分集成检查；不是真实测试结果',
            'temporary_directory':str(temp),'code_sha256':sha(production/'scripts/run_pubmed_geometry.py'),
            'conditions':len(result['summary'])}
    save(production/'artifacts/experiments/pubmed_geometry_v1/pipeline_smoke.json',record)
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
