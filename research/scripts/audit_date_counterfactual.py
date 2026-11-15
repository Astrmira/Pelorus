"""完成评分后的审计与描述性分层；不训练、不选模、不覆盖主结果。"""

import json
import numpy as np
import torch

from geometry_readouts import GeometryReadout
from prepare_date_counterfactual import EXP,ROOT,save,sha
from run_date_counterfactual import load,predict_ridge


def main():
    if (EXP/'posthoc_audit.json').exists():
        raise RuntimeError('本轮审计已保存，禁止覆盖')
    torch.set_num_threads(1)
    results=json.loads((EXP/'results.json').read_text(encoding='utf-8'))
    registry=json.loads((EXP/'selection_registry.json').read_text(encoding='utf-8'))
    heads=json.loads((EXP/'head_selection.json').read_text(encoding='utf-8'))
    protocol=json.loads((EXP/'protocol.json').read_text(encoding='utf-8'))
    checks={
        'protocol':sha(EXP/'protocol.json')==results['protocol_sha256'],
        'selection_registry':sha(EXP/'selection_registry.json')==results['selection_registry_sha256'],
        'head_selection':sha(EXP/'head_selection.json')==results['head_selection_sha256'],
        'scalar_checkpoint':sha(EXP/'scalar_heads.pt')==heads['checkpoint_sha256'],
    }
    checks.update({r['checkpoint']:sha(EXP/r['checkpoint'])==r['checkpoint_sha256'] for r in registry})
    checks.update({'code_'+n:sha(ROOT/'scripts'/n)==h for n,h in protocol['code_sha256'].items()})
    if not all(checks.values()):
        raise ValueError('冻结文件或检查点哈希不符')
    cases,corpus,_,_=load()
    device=corpus['queries'].device
    fitted=torch.load(EXP/'scalar_heads.pt',map_location=device,weights_only=True)
    strata=[]
    for split in ('interpolation','extrapolation'):
        names=sorted({r['name'] for r in results['runs'] if r['split']==split})
        for name in names:
            group=[r for r in results['runs'] if r['split']==split and r['name']==name]
            for stratum in ('same_year','cross_year'):
                selected=[[c for c in r['cases'] if c['stratum']==stratum] for r in group]
                strata.append({'split':split,'name':name,'stratum':stratum,'clusters':len(selected[0]),
                               **{m:float(np.mean([[c[m] for c in row] for row in selected]))
                                  for m in ('strict','paired','ordinary','temporal_single')}})
    flat=[]
    with torch.no_grad():
        for selected in registry:
            if selected['name']!='poincare_1':
                continue
            model=GeometryReadout('poincare_1',1536,8).to(device=device,dtype=torch.float64)
            model.load_state_dict(torch.load(EXP/selected['checkpoint'],map_location=device,weights_only=True)['selected'])
            for split in ('interpolation','extrapolation'):
                changed=total=0
                for c in cases:
                    if c['split']!=split:
                        continue
                    q=corpus['queries'][[r['qi'] for r in c['queries'] if r['style']!='neutral']]
                    docs=corpus['documents'][c['di']].reshape(2,2,-1)
                    for condition in (0,1):
                        candidates=docs[condition].unsqueeze(0).expand(len(q),-1,-1)
                        model.name='poincare_1'
                        hyper=model(q,candidates).argmax(-1)
                        model.name='shared_euclidean'
                        euclidean=model(q,candidates).argmax(-1)
                        changed+=int((hyper!=euclidean).sum())
                        total+=len(q)
                flat.append({'split':split,'seed':selected['seed'],'changed':changed,'total':total})
    diagnostic=[]
    for split in ('train','validation','interpolation','extrapolation'):
        subset=[c for c in cases if c['split']==split]
        confusion=np.zeros((3,3),dtype=int)
        order=[]
        family_roles={}
        date_errors=[]
        with torch.no_grad():
            for c in subset:
                queries=[q for q in c['queries'] if q['style']!='neutral']
                predictions=predict_ridge(fitted['query'],corpus['queries'][[q['qi'] for q in queries]]).argmax(-1).tolist()
                for q,predicted in zip(queries,predictions):
                    actual=q['role'] if q['style']=='temporal' else 2
                    confusion[actual,predicted]+=1
                for family in sorted({q['family'] for q in queries if q['style']=='temporal'}):
                    ix=[i for i,q in enumerate(queries) if q['style']=='temporal' and q['family']==family]
                    family_roles.setdefault(str(family),[]).append(all(predictions[i]==queries[i]['role'] for i in ix))
                dates=(predict_ridge(fitted['date'],corpus['documents'][c['di']])*365).tolist()
                date_errors.extend(abs(p-d['ordinal']) for p,d in zip(dates,c['documents']))
                order.append({'id':c['id'],'cluster':c['cluster'],'stratum':c['stratum'],
                              'true_ordinals':[d['ordinal'] for d in c['documents']],
                              'predicted_ordinals':dates,'strict_order':dates[0]<dates[1] and dates[2]>dates[3]})
        diagnostic.append({'split':split,'query_role_order':['earlier','later','ordinary'],
                           'confusion_true_rows_predicted_columns':confusion.tolist(),
                           'query_pair_role_accuracy_by_family':{k:float(np.mean(v)) for k,v in family_roles.items()},
                           'date_mae_days':float(np.mean(date_errors)),'date_cases':order})
    oracle=[r for r in results['summary'] if r['name']=='both_oracles']
    checks['oracle_temporal_definition']=all(r[m]['mean']==1. for r in oracle for m in ('strict','paired','temporal_single'))
    if not checks['oracle_temporal_definition']:
        raise ValueError('真日期加真查询类别未通过定义自检')
    save(EXP/'posthoc_audit.json',{'stage':'主结果完成后的描述性审计，未进行训练或选择',
                                 'script_sha256':sha(ROOT/'scripts/audit_date_counterfactual.py'),
                                 'results_sha256':sha(EXP/'results.json'),'checks':checks,'strata':strata,
                                 'fixed_coordinates_flat_prediction_changes':flat,'head_diagnostic':diagnostic})
    print('冻结文件、检查点和标签定义核验通过，已保存分层诊断与固定坐标消融审计。',flush=True)


if __name__=='__main__':
    main()
