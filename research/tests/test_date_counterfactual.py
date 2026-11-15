"""检验受控日期的隔离、反事实目标与诊断探针。"""

import json
import sys
from datetime import date
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_date_counterfactual import DATA
from run_date_counterfactual import summarize_case,fit_ridge,predict_ridge


def cases():return json.loads((DATA/'annotations.json').read_text(encoding='utf-8'))


def test_dates_and_subjects_are_held_out():
    rows=cases()
    groups={name:[c for c in rows if c['split']==name] for name in ('train','validation','interpolation','extrapolation')}
    assert [len(groups[name]) for name in groups]==[48,12,24,24]
    dates={name:{d['date'] for c in cs for d in c['documents']} for name,cs in groups.items()}
    for i,a in enumerate(groups):
        for b in list(groups)[i+1:]:assert dates[a].isdisjoint(dates[b])
    train_subjects={c['subject'] for c in groups['train']}
    assert train_subjects.isdisjoint({c['subject'] for c in groups['validation']+groups['interpolation']+groups['extrapolation']})
    assert {c['subject'] for c in groups['interpolation']}=={c['subject'] for c in groups['extrapolation']}
    assert all(date.fromisoformat(d).year in (2024,2025) for d in dates['extrapolation'])
    assert all((date.fromisoformat(c['dates'][1])-date.fromisoformat(c['dates'][0])).days>=30 for c in rows)


def test_date_intervention_keeps_body_and_fact_positions():
    docs={r['id']:r['text'] for r in map(json.loads,(DATA/'documents.jsonl').read_text(encoding='utf-8').splitlines())}
    for c in cases():
        d=[docs[r['id']] for r in c['documents']]
        assert d[0].split('\n',1)[1]==d[2].split('\n',1)[1]
        assert d[1].split('\n',1)[1]==d[3].split('\n',1)[1]
        assert d[0].split('\n',1)[0]==d[3].split('\n',1)[0]
        assert d[1].split('\n',1)[0]==d[2].split('\n',1)[0]
        assert [r['colour'] for r in c['documents']]==['navy','scarlet','navy','scarlet']


def test_counterfactual_score_requires_both_opposite_targets():
    c=cases()[0];queries=c['queries'];s=torch.zeros(2,len(queries),2,dtype=torch.float64)
    for k in (0,1):
        for i,q in enumerate(queries):s[k,i,q['role']^k if q['style']=='temporal' else q['role']]=1
    m=summarize_case(s,c)
    assert m['strict']==m['paired']==m['ordinary']==1
    s[:,:,0]=1;s[:,:,1]=0
    m=summarize_case(s,c)
    assert m['strict']==m['paired']==0 and m['ordinary']==.5


def test_ridge_recovers_simple_numeric_relation_without_test_refit():
    x=torch.tensor([[-2.,0.],[-1.,1.],[0.,-1.],[1.,1.],[2.,0.]],dtype=torch.float64)
    y=x@torch.tensor([3.,-2.],dtype=torch.float64)+10
    fit=fit_ridge(x,y,1e-8)
    new=torch.tensor([[.2,.7],[-.3,.1]],dtype=torch.float64)
    torch.testing.assert_close(predict_ridge(fit,new),new@torch.tensor([3.,-2.],dtype=torch.float64)+10,rtol=1e-6,atol=1e-6)
