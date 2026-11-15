"""检验新终检的数据身份、相反目标、反事实干预及主题簇统计。"""

import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_geometry_confirmatory import ROOT, DATA, EXP, DEV
from evaluate_geometry_confirmatory import cluster_interval, score_case
from geometry_readouts import GeometryReadout
from run_geometry_controlled import sha


def cases():
    return json.loads((DATA/'annotations.json').read_text(encoding='utf-8'))


def test_new_sources_are_disjoint_and_have_no_shared_old_evidence():
    new = cases()
    old = []
    for name in ('temporal_readout','temporal_fresh_test'):
        old += json.loads((ROOT/'data/processed'/name/'annotations.json').read_text(encoding='utf-8'))
    assert {c['id'] for c in new}.isdisjoint({c['id'] for c in old})
    old_sessions = {e.get('session_fingerprint') for c in old for e in c['evidence']} - {None}
    assert not any(e['session_fingerprint'] in old_sessions for c in new for e in c['evidence'])
    assert len(new)==50 and len({c['evidence_group'] for c in new})==50
    assert len({c['topic'] for c in new})==26


def test_ordinary_spans_and_record_targets_are_unambiguous():
    for c in cases():
        assert c['evidence'][0]['date'] < c['evidence'][1]['date']
        for i,e in enumerate(c['evidence']):
            assert e['user_count']==1
            assert e['ordinary_value'].lower() in e['body'].lower()
            assert e['ordinary_value'].lower() not in c['evidence'][1-i]['body'].lower()
        record = [q for q in c['queries'] if q['style']=='record']
        assert len(record)==6
        for f in (0,1,2):
            assert [q['target_index'] for q in record if q['family']==f]==[0,1]


def test_only_metadata_changes_under_date_swap():
    documents = {r['id']:r['text'] for r in [json.loads(l) for l in (DATA/'documents.jsonl').read_text(encoding='utf-8').splitlines()]}
    for c in cases():
        original,swapped,removed = ([documents[x] for x in c['document_variants'][v]] for v in ('dated','swapped_dates','no_date'))
        for i in (0,1):
            assert original[i].split('\n',1)[1]==swapped[i].split('\n',1)[1]==removed[i]
            assert original[i].split('\n',1)[0]==swapped[1-i].split('\n',1)[0]


def test_event_labels_can_disagree_with_record_dates():
    events = [c for c in cases() if 'event_review' in c]
    assert len(events)==14
    assert any(c['event_review']['earlier_index']==1 for c in events)
    for c in events:
        for family in (0,1,2):
            pair=[q for q in c['queries'] if q['style']=='event' and q['family']==family]
            assert [q['target_index'] for q in pair]==[c['event_review']['earlier_index'],1-c['event_review']['earlier_index']]


def test_protocol_and_models_match_preencoding_freeze():
    p=json.loads((EXP/'protocol.json').read_text(encoding='utf-8'))
    assert p['development_registry_sha256']==sha(DEV/'selection_registry.json')
    for name,h in p['checkpoints_sha256'].items():
        assert sha(DEV/name)==h
    assert p['input_manifest_sha256']==sha(DATA/'manifest.json')


def test_cluster_bootstrap_does_not_count_templates_as_samples():
    result=cluster_interval([1.,1.,0.],['同主题','同主题','其他'],replicates=1000)
    assert result['clusters']==2 and result['chains']==3
    assert np.isclose(result['estimate'],2/3)
    constant=cluster_interval([.1,.1,.1],['甲','甲','乙'],replicates=1000)
    np.testing.assert_allclose(constant['ci95'],[.1,.1])


def test_unchanged_scores_cannot_pass_strict_date_following():
    case=cases()[0]
    qids=[q['id'] for q in case['queries']]
    all_ids=sum(case['document_variants'].values(),[])
    docs=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64).repeat(3,1)
    corpus={'queries_index':{v:i for i,v in enumerate(qids)},'documents_index':{v:i for i,v in enumerate(all_ids)},
            'queries':torch.tensor([[1.,0.]],dtype=torch.float64).repeat(len(qids),1),'documents':docs}
    result=score_case(GeometryReadout('raw_cosine',2).double(),case,corpus)
    assert result['neutral_paired']==0 and result['strict_date_following']==0
    assert result['dated_record']['paired']==0 and result['swapped_record']['paired']==0
