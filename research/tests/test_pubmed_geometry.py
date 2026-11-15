"""检验真实语料实验的数据隔离、保守排名和几何交叉比较。"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_pubmed_geometry import build,write_jsonl
from encode_embeddings_api import read_jsonl
from run_pubmed_geometry import Readout,ranks,paired_interval,assemble


def toy_records():
    records=[];descriptors={}
    for topic in range(40):
        stem=f'C01.100.{topic:03d}'
        for suffix,count in (('.001.001',3),('.001.002',3),('.001',2),('',2)):
            tree=stem+suffix;uid='D'+tree
            descriptors[uid]={'name':uid,'trees':[tree]}
            for number in range(count):
                pmid=f'{topic}_{suffix}_{number}'
                records.append({'pmid':pmid,'title':'标题'+pmid,'abstract':'摘要'+pmid,
                                'mesh':[{'id':uid,'major':True}]})
    return records,descriptors


def test_partition_and_candidates():
    records,descriptors=toy_records()
    docs,queries,_=build(records,descriptors,pool_size=len(records),counts={'train':20,'validation':10,'test':20})
    index={r['id']:r for r in docs}
    topics={s:{q['cluster'] for q in queries if q['split']==s} for s in ('train','validation','test')}
    assert not topics['train']&topics['validation']
    assert not topics['train']&topics['test']
    assert not topics['validation']&topics['test']
    for q in queries:
        assert len(q['candidates'])==10
        assert len({d['id'] for d in q['candidates']})==10
        assert q['candidates'][q['target_index']]['id']==q['target_id']
        parent=q['anchor_tree'].rsplit('.',1)[0]
        for d in q['candidates']:
            tree=index[d['id']]['anchor_tree']
            if d['type']=='parent':
                assert tree==parent
            if d['type']=='grandparent':
                assert tree==parent.rsplit('.',1)[0]
            if d['type']=='sibling':
                assert tree!=q['anchor_tree'] and tree.rsplit('.',1)[0]==parent


def test_ties_do_not_reward_target_position():
    score=torch.tensor([[1.,1.,1.],[3.,2.,1.],[1.,2.,3.]])
    actual=torch.tensor([0,0,0])
    rank,_=ranks(score,actual)
    assert rank.tolist()==[3,1,3]


def test_small_batch_candidate_index_is_two_dimensional():
    vectors={'queries':torch.randn(3,1536),'documents':torch.randn(20,1536)}
    queries=[{'qi':0,'di':list(range(10)),'target_index':0},
             {'qi':2,'di':list(range(10,20)),'target_index':9}]
    q,d,y=assemble(queries,vectors)
    assert q.shape==(2,1536) and d.shape==(2,10,1536)
    assert y.tolist()==[0,9]
    assert torch.equal(d[1,9],vectors['documents'][19])


def test_real_text_unicode_separators_roundtrip(tmp_path):
    rows=[{'id':'文章1','text':'标题\u2028换行\u2029分段\n摘要\x85继续'}]
    path=tmp_path/'documents.jsonl'
    write_jsonl(path,rows)
    assert read_jsonl(path)==rows


def test_equal_capacity_and_identical_initial_coordinates():
    models=[]
    for name in ('shared_cosine','shared_euclidean','poincare_1','nonlinear_euclidean'):
        torch.manual_seed(11)
        models.append(Readout(name).double())
    assert len({sum(p.numel() for p in m.parameters()) for m in models})==1
    x=torch.randn(7,1536,dtype=torch.float64)
    assert torch.equal(models[1].coordinates(x),models[2].coordinates(x))
    models[2].name='shared_euclidean'
    d=torch.randn(7,10,1536,dtype=torch.float64)
    assert torch.allclose(models[1](x,d),models[2](x,d),atol=1e-12)


def test_seed_repeats_are_not_independent_topics():
    def run(name,seed,delta):
        return {'name':name,'seed':seed,'local':[{'cluster':f'主题{i}','r1':float(i%2)+delta} for i in range(8)]}
    single=[run('a',1,.1),run('b',1,0)]
    repeated=[run(n,s,d) for n,d in (('a',.1),('b',0)) for s in range(5)]
    a=paired_interval(single,'a','b');b=paired_interval(repeated,'a','b')
    assert a['topics']==b['topics']==8
    np.testing.assert_allclose(a['difference'],b['difference'])
    np.testing.assert_allclose(a['ci975'],b['ci975'])
