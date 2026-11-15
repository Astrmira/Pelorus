"""复核冻结标注，并在临时合成向量上检查完整选择及评分流程。"""

import json
from pathlib import Path
import tempfile
import numpy as np
import torch

import prepare_mesh_intent as p
import run_mesh_intent as m


def main():
    manifest=json.loads((p.DATA/'manifest.json').read_text(encoding='utf-8'))
    entries=p.parse_mesh(p.SOURCE.read_bytes())
    docs=json.loads((p.DATA/'document_metadata.json').read_text(encoding='utf-8'))
    focus=json.loads((p.DATA/'focus_metadata.json').read_text(encoding='utf-8'))
    queries=json.loads((p.DATA/'annotations.json').read_text(encoding='utf-8'))
    assert p.sha(p.SOURCE)==manifest['source_sha256']
    assert all(p.sha(p.DATA/name)==value for name,value in manifest['input_sha256'].items())
    assert p.validate(entries,docs,focus,queries)
    for r in docs:
        assert r['definition']==entries[r['id']]['definition']
        assert r['parents']==entries[r['id']]['parents']
    examples=json.loads((p.DATA/'review_examples.json').read_text(encoding='utf-8'))
    assert len(examples)==8 and all(r['focus']['split']!='test' for r in examples)
    for r in examples:
        print('抽查：'+r['focus']['split']+'；'+r['focus']['name']+' → '+', '.join(x['name'] for x in r['parents']),flush=True)
    p.save(p.EXP/'data_review.json',{'passed':True,'manifest_sha256':p.sha(p.DATA/'manifest.json'),
           'checks':['官方源文件哈希一致','全部输入哈希一致','全部直接父标签一致','全部疾病分支隔离',
                     '留出焦点未出现在较早划分正例','全部文档定义与官方源一致','八个训练或验证例结构抽查'],
           'scope':'程序标注与结构复核，未进行独立医学专家评审；未查看测试检索结果。'})
    production_exp=p.EXP
    lock=json.loads((p.EXP/'encoding_lock.json').read_text(encoding='utf-8'))
    # 所有合成数据及检查点均位于独立临时目录，绝不覆盖正式数据或向量。
    with tempfile.TemporaryDirectory(prefix='quiver-mesh-smoke-') as directory:
        root=Path(directory)
        m.DATA=root/'data';m.EMBED=root/'embeddings';m.EXP=root/'experiment'
        for folder in (m.DATA,m.EMBED,m.EXP):
            folder.mkdir()
        m.DIM=8;m.SEEDS=(11,);m.LRS=(.0005,);m.STEPS=2
        p.save(m.DATA/'manifest.json',{'scope':'仅供流程检查的合成向量'})
        p.save(m.EXP/'encoding_lock.json',{**lock,'manifest_sha256':p.sha(m.DATA/'manifest.json')})
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rng=np.random.default_rng(20261016)
        documents=rng.normal(size=(100,1536))
        documents/=np.linalg.norm(documents,axis=1,keepdims=True)
        ids=[f'd{i}' for i in range(len(documents))]
        records=[];vectors=[]
        for i in range(24):
            split='train' if i<16 else ('validation' if i<20 else 'test')
            for family in (0,1):
                for intent in ('self','parent'):
                    target=i if intent=='self' else 50+i
                    positives=[target]
                    if intent=='parent' and i%2:
                        positives.append(75+i)
                    records.append({'id':f'synthetic_{i}_{family}_{intent}','qi':len(vectors),
                                    'focus_id':ids[i],'cluster':f'branch{i}','family':family,
                                    'split':split,'intent':intent,'positive_di':positives,
                                    'positive_ids':[ids[j] for j in positives]})
                    vectors.append(documents[target])
        vectors=np.stack(vectors)
        np.save(m.EMBED/'documents.npy',documents);np.save(m.EMBED/'queries.npy',vectors)
        p.save(m.EMBED/'runtime_manifest.json',{'scope':'仅供流程检查，不调用编码API'})
        corpus={}
        for name,array in (('documents',documents),('queries',vectors)):
            corpus[name]=torch.tensor(array,dtype=torch.float64,device=device)
            corpus['raw_'+name]=corpus[name]
        basis=torch.tensor(np.linalg.qr(rng.normal(size=(1536,8)))[0].T,dtype=torch.float64,device=device)
        center=torch.zeros(1536,dtype=torch.float64,device=device)
        m.load=lambda:(records,corpus,ids,center,basis,np.zeros(1536),np.ones(1536),list(range(16)))
        assert m.execute() is True
        results=json.loads((m.EXP/'results.json').read_text(encoding='utf-8'))
        registry=json.loads((m.EXP/'selection_registry.json').read_text(encoding='utf-8'))
        runs=json.loads((m.EXP/'training_runs.json').read_text(encoding='utf-8'))
        assert results['test_scored'] and all(results['checks'].values())
        assert len(registry['selected'])==3 and not registry['failures'] and len(runs)==3
        assert len(results['summary'])==7
        assert all(r['macro']['self']==1 for r in results['summary'])
        p.save(production_exp/'smoke_verification.json',{'passed':True,'code_sha256':lock['code_sha256'],
               'scope':'仅在临时合成向量上完成三种几何训练、验证选择、检查点冻结、评分与曲率交叉评分；不构成真实任务研究证据。',
               'training_runs':len(runs),'selected_checkpoints':len(registry['selected']),
               'score_groups':len(results['summary']),'device':str(device),'production_test_scored':False})
    print('冻结数据复核及独立合成流程检查均通过，正式测试仍未评分。',flush=True)


if __name__=='__main__':
    main()
