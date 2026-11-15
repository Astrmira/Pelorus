"""验证诊断的数据隔离、无监督拟合范围及全库梯度。"""

import sys
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_pubmed_readout import restrict_queries,guard_queries,pca_fit,CosineHead,score_matrix,row_metrics


def test_test_queries_are_filtered_and_rejected():
    rows=[{'split':s,'id':s} for s in ('train','validation','test')]
    assert [r['id'] for r in restrict_queries(rows)]==['train','validation']
    with pytest.raises(ValueError,match='禁止评分测试'):
        guard_queries(rows)
    with pytest.raises(ValueError,match='未知'):
        restrict_queries([{'split':'hidden'}])


def test_pca_ignores_documents_outside_fit_indices():
    a=torch.tensor([[-3.,0.],[0.,0.],[3.,0.],[0.,1.]],dtype=torch.float64)
    c1,b1=pca_fit(a,[0,1,2])
    a[3]=torch.tensor([1e6,1e8])
    c2,b2=pca_fit(a,[0,1,2])
    assert torch.equal(c1,c2)
    assert torch.allclose(b1[:1].T@b1[:1],b2[:1].T@b2[:1])
    assert torch.allclose(b2[:1].T@b2[:1],torch.diag(torch.tensor([1.,0.],dtype=torch.float64)))


def test_full_softmax_propagates_to_nonlocal_negative():
    head=CosineHead(torch.eye(2,dtype=torch.float64),torch.zeros(2,dtype=torch.float64))
    q=torch.tensor([[1.,.2]],dtype=torch.float64)
    docs=torch.tensor([[1.,0.],[0.,1.],[-1.,1.]],dtype=torch.float64,requires_grad=True)
    loss=F.cross_entropy(score_matrix(head,q,docs),torch.tensor([0]))
    loss.backward()
    assert docs.grad[2].norm()>0
    assert head.weight.grad.norm()>0


def test_initial_pca_head_preserves_pca_cosine():
    center=torch.tensor([.2,.1,0.],dtype=torch.float64)
    basis=torch.tensor([[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
    q=torch.tensor([[1.,2.,3.]],dtype=torch.float64)
    d=torch.tensor([[1.,0.,1.],[0.,1.,1.]],dtype=torch.float64)
    expected=F.normalize((q-center)@basis.T,dim=-1)@F.normalize((d-center)@basis.T,dim=-1).T
    assert torch.allclose(score_matrix(CosineHead(basis,center),q,d),expected)


def test_ranking_rejects_test_and_treats_ties_conservatively():
    with pytest.raises(ValueError,match='禁止评分测试'):
        row_metrics(torch.ones(1,3),[{'split':'test','target_di':0}])
    rows=[{'id':'q','cluster':'c','split':'validation','target_di':0}]
    metric=row_metrics(torch.ones(1,3),rows)[0]
    assert metric['rank']==3 and metric['r1']==0 and metric['rr']==1/3
