"""检验全部父类标签、疾病分支隔离、精确距离与普通检索门槛。"""

import gzip
import sys
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_mesh_intent import parse_mesh,sample_data,validate
from run_mesh_intent import geometry_scores,positive_loss,IntentReadout,case_metrics,selection_key,evaluate,METRICS
from geometry_readouts import tangent_distance_squared


def test_all_tree_paths_contribute_direct_parents():
    def entry(uid,trees):
        return f'<DescriptorRecord><DescriptorUI>{uid}</DescriptorUI><DescriptorName><String>{uid}</String></DescriptorName><TreeNumberList>'+''.join(f'<TreeNumber>{t}</TreeNumber>' for t in trees)+'</TreeNumberList><ConceptList><Concept PreferredConceptYN="Y"><ScopeNote>Official definition of a biomedical concept.</ScopeNote></Concept></ConceptList></DescriptorRecord>'
    xml='<DescriptorRecordSet>'+entry('p1',['C01.100.001'])+entry('p2',['C02.100.001'])+entry('c',['C01.100.001.001','C02.100.001.001'])+'</DescriptorRecordSet>'
    records=parse_mesh(gzip.compress(xml.encode()))
    assert records['c']['parents']==['p1','p2']


def toy_entries():
    entries={}
    for i in range(40):
        stem=f'C01.100.{i:03d}'
        definitions=(('g',[stem],[]),('p1',[stem+'.001'],['g']),('p2',[stem+'.002'],['g']),
                     ('c1',[stem+'.001.001'],['p1']),('c2',[stem+'.001.002',stem+'.002.002'],['p1','p2']))
        for suffix,paths,parents in definitions:
            uid=f'{i}_{suffix}'
            entries[uid]={'id':uid,'name':uid,'definition':'真实专家定义示例。'*15,'trees':paths,
                          'parents':[f'{i}_{p}' for p in parents],'branches':[stem]}
    return entries


def test_data_partition_and_multi_parent_coverage():
    entries=toy_entries()
    docs,focus,queries,_=sample_data(entries,counts={'train':20,'validation':8,'test':16},pool_size=len(entries))
    assert validate(entries,docs,focus,queries)
    for split,expected in (('train',5),('validation',2),('test',4)):
        assert sum(r['split']==split and len(r['parents'])>1 for r in focus)==expected
    q=next(q for q in queries if q['intent']=='parent' and len(q['positive_ids'])==2)
    old=q['positive_ids'];q['positive_ids']=old[:1]
    with pytest.raises(ValueError,match='完整保留'):
        validate(entries,docs,focus,queries)


@pytest.mark.parametrize('mode',['euclidean','hyperbolic'])
def test_gram_distance_and_gradients_match_exact_pairwise(mode):
    torch.manual_seed(11)
    q=(torch.randn(3,5,dtype=torch.float64)*.6).requires_grad_()
    d=(torch.randn(7,5,dtype=torch.float64)*.6).requires_grad_()
    curvature=1. if mode=='hyperbolic' else 0.
    expected=-tangent_distance_squared(q[:,None,:],d[None,:,:],curvature)
    actual=geometry_scores(q,d,mode)
    assert torch.allclose(actual,expected,atol=1e-11,rtol=1e-11)
    g1=torch.autograd.grad(actual.sum(),(q,d),retain_graph=True)
    g2=torch.autograd.grad(expected.sum(),(q,d))
    assert all(torch.allclose(a,b,atol=1e-10,rtol=1e-10) for a,b in zip(g1,g2))


def test_all_positives_receive_loss_supervision():
    scores=torch.tensor([[.2,.1,-.1,.3]],dtype=torch.float64,requires_grad=True)
    indices=torch.tensor([[1,2]]);valid=torch.tensor([[True,True]])
    loss=positive_loss(scores,indices,valid)
    expected=-F.log_softmax(scores,dim=-1)[0,[1,2]].mean()
    assert torch.allclose(loss,expected)
    loss.backward()
    assert scores.grad[0,1]<0 and scores.grad[0,2]<0 and scores.grad[0,0]>0


def test_any_true_parent_and_negative_ties():
    queries=[{'id':'s','focus_id':'c','cluster':'b','family':0,'intent':'self','positive_di':[0],'positive_ids':['c']},
             {'id':'p','focus_id':'c','cluster':'b','family':0,'intent':'parent','positive_di':[1,2],'positive_ids':['p1','p2']}]
    scores=torch.tensor([[5.,0.,0.,0.],[4.,0.,5.,0.]],dtype=torch.float64)
    cases=case_metrics(scores,queries,['c','p1','p2','n'])
    assert cases[0]['pair']==1 and cases[0]['multi_parent']
    scores[1,3]=5
    assert case_metrics(scores,queries,['c','p1','p2','n'])[0]['pair']==0


def test_equal_capacity_and_identical_pca_initialization():
    basis=torch.randn(4,6,dtype=torch.float64);center=torch.zeros(6,dtype=torch.float64)
    models=[IntentReadout(mode,basis,center) for mode in ('cosine','euclidean','hyperbolic')]
    assert {sum(p.numel() for p in m.parameters()) for m in models}=={48}
    assert all(torch.equal(m.query_weight,m.document_weight) for m in models)
    assert all(torch.equal(m.query_weight,models[0].query_weight) for m in models)


def test_retention_gate_and_blind_test_guard():
    case={'id':'c','cluster':'b',**{m:1. for m in METRICS}}
    case['self']=.9
    assert selection_key([case],.98) is None
    case['self']=.98
    assert selection_key([case],.98) is not None
    with pytest.raises(ValueError,match='禁止测试评分'):
        evaluate(None,[{'split':'test'}],{},[],False)
