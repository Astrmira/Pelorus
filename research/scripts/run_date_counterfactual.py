"""在新日期反事实文本上诊断冻结向量、查询条件与曲率的作用。"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from geometry_readouts import GeometryReadout
from prepare_date_counterfactual import DATA,EMBED,EXP,ROOT,save,sha

MODELS=('shared_cosine','shared_euclidean','poincare_1','bilinear','nonlinear_euclidean')
SEEDS=(11,23,47)
LRS=(.0005,.005)
STEPS=800
ALPHAS=(.0001,.001,.01,.1,1.)


def load():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['input_sha256'].items():
        if sha(DATA/name)!=h:raise ValueError('日期反事实输入发生变化')
    runtime=json.loads((EMBED/'runtime_manifest.json').read_text(encoding='utf-8'))
    if runtime['request_model']!='Qwen3-Embedding-8B' or runtime['dimensions']!=1536:raise ValueError('向量模型或维度不符')
    for kind in ('documents','queries'):
        if runtime['input_sha256'][kind]!=manifest['input_sha256'][kind+'.jsonl']:raise ValueError('向量文本哈希不符')
    cases=json.loads((DATA/'annotations.json').read_text(encoding='utf-8'))
    arrays={k:np.load(EMBED/(k+'.npy')) for k in ('documents','queries')}
    indices={k:{v:i for i,v in enumerate(json.loads((EMBED/(('document' if k=='documents' else 'query')+'_ids.json')).read_text(encoding='utf-8')))} for k in arrays}
    train_docs=[indices['documents'][d['id']] for c in cases if c['split']=='train' for d in c['documents']]
    mu=arrays['documents'][train_docs].mean(0)
    std=np.maximum(arrays['documents'][train_docs].std(0),1e-3)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus={}
    for k,v in arrays.items():
        if not np.isfinite(v).all() or np.any(np.linalg.norm(v,axis=1)==0):raise ValueError('向量非有限或为零')
        corpus['raw_'+k]=torch.as_tensor(v/np.linalg.norm(v,axis=1,keepdims=True),dtype=torch.float64,device=device)
        w=(v-mu)/std
        corpus[k]=torch.as_tensor(w/np.linalg.norm(w,axis=1,keepdims=True),dtype=torch.float64,device=device)
    for c in cases:
        c['di']=[indices['documents'][d['id']] for d in c['documents']]
        for q in c['queries']:q['qi']=indices['queries'][q['id']]
    return cases,corpus,mu,std


def assemble(cases,corpus):
    q,d,y=[],[],[]
    for c in cases:
        for condition,docs in enumerate((c['di'][:2],c['di'][2:])):
            for query in c['queries']:
                if query['style']=='neutral':continue
                q.append(query['qi']);d.append(docs)
                y.append(query['role']^condition if query['style']=='temporal' else query['role'])
    return corpus['queries'][q],corpus['documents'][d],torch.tensor(y,device=corpus['queries'].device)


def summarize_case(scores,case):
    queries=case['queries']
    margins=[]
    for condition in (0,1):
        targets=torch.tensor([q['role']^condition if q['style']=='temporal' else q['role'] for q in queries],device=scores.device)
        s=scores[condition]
        margins.append((s.gather(1,targets[:,None])-s.gather(1,(1-targets)[:,None])).squeeze(1).tolist())
    families=sorted({q['family'] for q in queries if q['style']=='temporal'})
    strict,paired=[],[]
    for family in families:
        ix=[i for i,q in enumerate(queries) if q['style']=='temporal' and q['family']==family]
        strict.append(all(margins[k][i]>1e-6 for k in (0,1) for i in ix))
        paired.extend(all(margins[k][i]>1e-6 for i in ix) for k in (0,1))
    ordinary=[margins[k][i]>1e-6 for k in (0,1) for i,q in enumerate(queries) if q['style']=='ordinary']
    return {'id':case['id'],'cluster':case['cluster'],'stratum':case['stratum'],
            'strict':float(np.mean(strict)),'paired':float(np.mean(paired)),'ordinary':float(np.mean(ordinary)),
            'temporal_single':float(np.mean([margins[k][i]>1e-6 for k in (0,1) for i,q in enumerate(queries) if q['style']=='temporal']))}


def evaluate(model,corpus,cases,scale=1.,raw=False):
    prefix='raw_' if raw else ''
    rows=[]
    with torch.no_grad():
        for c in cases:
            q=corpus[prefix+'queries'][[x['qi'] for x in c['queries']]]
            docs=corpus[prefix+'documents'][c['di']].reshape(2,2,-1)
            scores=torch.stack([model(q,docs[k].unsqueeze(0).expand(len(q),-1,-1))/scale for k in (0,1)])
            if not torch.isfinite(scores).all():raise ValueError('评分包含非有限值')
            rows.append(summarize_case(scores,c))
    return rows


def key(rows):
    return (np.mean([r['strict'] for r in rows]),np.mean([r['paired'] for r in rows]),np.mean([r['ordinary'] for r in rows]))


def fit(name,seed,lr,corpus,train,val):
    torch.manual_seed(seed)
    model=GeometryReadout(name,1536,8).to(device=corpus['queries'].device,dtype=torch.float64)
    x,d,y=assemble(train,corpus)
    with torch.no_grad():
        s=model(x,d);scale=float((s[:,0]-s[:,1]).square().mean().sqrt().clamp_min(.01))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best=evaluate(model,corpus,val,scale);selected=copy.deepcopy(model.state_dict());chosen=0
    rng=np.random.default_rng(seed);history=[]
    for step in range(1,STEPS+1):
        ix=torch.tensor(rng.choice(len(x),64,replace=False),device=x.device)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[ix],d[ix])/scale,y[ix])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%50==0:
            metrics=evaluate(model,corpus,val,scale)
            history.append({'step':step,'validation_key':tuple(float(v) for v in key(metrics)),'train_loss':float(loss.detach())})
            if key(metrics)>key(best):best,selected,chosen=metrics,copy.deepcopy(model.state_dict()),step
    last=evaluate(model,corpus,train,scale)
    model.load_state_dict(selected)
    return model,{'name':name,'seed':seed,'lr':lr,'selected_step':chosen,'scale':scale,'validation':best,'history':history,
                  'last_train_key':tuple(float(v) for v in key(last)),'parameters':sum(p.numel() for p in model.parameters())}


def fit_ridge(x,y,alpha):
    center=x.mean(0);intercept=y.mean(0)
    xc=x-center
    weights=xc.T@torch.linalg.solve(xc@xc.T+alpha*torch.eye(len(x),device=x.device,dtype=x.dtype),y-intercept)
    return {'center':center,'weights':weights,'intercept':intercept}


def predict_ridge(fit,x):
    return (x-fit['center'])@fit['weights']+fit['intercept']


def fit_heads(corpus,train,val):
    def document_samples(cases):
        ix=[i for c in cases for i in c['di']]
        y=torch.tensor([d['ordinal'] for c in cases for d in c['documents']],device=corpus['queries'].device,dtype=torch.float64)
        return corpus['documents'][ix],y
    def query_samples(cases):
        queries=[q for c in cases for q in c['queries'] if q['style']!='neutral']
        y=torch.tensor([q['role'] if q['style']=='temporal' else 2 for q in queries],device=corpus['queries'].device)
        return corpus['queries'][[q['qi'] for q in queries]],y
    xd,yd=document_samples(train);vd,vy=document_samples(val)
    xq,yq=query_samples(train);vq,vqy=query_samples(val)
    candidates=[];best_date=best_query=None
    for a in ALPHAS:
        date_head=fit_ridge(xd,yd/365.,a)
        mae=float((predict_ridge(date_head,vd)*365-vy).abs().mean())
        query_head=fit_ridge(xq,F.one_hot(yq,3).to(torch.float64),a)
        accuracy=float((predict_ridge(query_head,vq).argmax(-1)==vqy).double().mean())
        candidates.append({'alpha':a,'validation_date_mae_days':mae,'validation_query_accuracy':accuracy})
        if best_date is None or mae<best_date[0]:best_date=(mae,a,date_head)
        if best_query is None or accuracy>best_query[0]:best_query=(accuracy,a,query_head)
    return best_date[2],best_query[2],{'candidates':candidates,'date_alpha':best_date[1],'query_alpha':best_query[1],
                                     'date_validation_mae_days':best_date[0],'query_validation_accuracy':best_query[0]}


def evaluate_heads(date_head,query_head,corpus,cases,oracle_date=False,oracle_query=False):
    rows=[];errors=[];query_correct=[]
    with torch.no_grad():
        for c in cases:
            q=corpus['queries'][[r['qi'] for r in c['queries']]]
            predicted_roles=predict_ridge(query_head,q).argmax(-1)
            known_roles=torch.tensor([r['role'] if r['style']=='temporal' else 2 for r in c['queries']],device=q.device)
            ix=[i for i,r in enumerate(c['queries']) if r['style']!='neutral']
            query_correct.extend((predicted_roles[ix]==known_roles[ix]).tolist())
            roles=known_roles if oracle_query else predicted_roles
            docs=corpus['documents'][c['di']]
            known_dates=torch.tensor([r['ordinal'] for r in c['documents']],device=q.device,dtype=q.dtype)
            predicted_dates=predict_ridge(date_head,docs)*365.
            errors.extend((predicted_dates-known_dates).abs().tolist())
            times=known_dates if oracle_date else predicted_dates
            raw_q=corpus['raw_queries'][[r['qi'] for r in c['queries']]]
            raw_d=corpus['raw_documents'][c['di']].reshape(2,2,-1)
            scores=[]
            for condition in (0,1):
                raw_score=(raw_q.unsqueeze(1)*raw_d[condition].unsqueeze(0)).sum(-1)
                t=times[2*condition:2*condition+2]
                s=torch.where((roles==0)[:,None],-t[None,:],torch.where((roles==1)[:,None],t[None,:],raw_score))
                scores.append(s)
            rows.append(summarize_case(torch.stack(scores),c))
    return {'cases':rows,'date_mae_days':float(np.mean(errors)),'query_role_accuracy':float(np.mean(query_correct))}


def bootstrap(values):
    a=np.asarray(values);rng=np.random.default_rng(20261016)
    draws=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return {'mean':float(a.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'clusters':len(a)}


def main():
    if (EXP/'selection_registry.json').exists():raise RuntimeError('本轮选择已冻结，禁止重复训练/选模')
    encoding_lock=json.loads((EXP/'encoding_lock.json').read_text(encoding='utf-8'))
    if encoding_lock['manifest_sha256']!=sha(DATA/'manifest.json'):
        raise ValueError('编码前冻结的数据清单发生变化')
    for name,expected in encoding_lock['code_sha256'].items():
        if sha(ROOT/'scripts'/name)!=expected:raise ValueError('编码前冻结的模型或评估代码发生变化')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cases,corpus,mu,std=load()
    splits={name:[c for c in cases if c['split']==name] for name in ('train','validation','interpolation','extrapolation')}
    protocol={'models':MODELS,'seeds':SEEDS,'lrs':LRS,'steps':STEPS,'batch':64,'dtype':'float64','device':str(corpus['queries'].device),
              'primary':'固定K=-1双曲减同参数欧氏的严格日期反转正确率；插值和外推分开，差值区间各97.5%（两个主要比较Bonferroni）',
              'selection':'只按验证严格反转、成对、普通单问依次选学习率/检查点；两头探针的alpha分别按验证日期MAE与查询类别准确率选择。',
              'heads':'日期监督的标量ridge + 三类别查询ridge；普通事实使用Raw Cosine回退。监督比检索读出更强，只作可读性诊断。',
              'freeze_before_test':True,'old_test_reuse':False,'manifest_sha256':sha(DATA/'manifest.json'),
              'code_sha256':{n:sha(ROOT/'scripts'/n) for n in ('run_date_counterfactual.py','geometry_readouts.py')},
              'input_vectors_sha256':{n:sha(EMBED/n) for n in ('documents.npy','queries.npy','runtime_manifest.json')},
              'limitations':['程序生成短文本，不能外推真实纵向记忆。','两个测试集共享24个物品主题，分别分析，不合并成48个主题。',
                             '标量两头探针使用额外日期与查询类别标签，不能将其提升归因于几何。']}
    save(EXP/'protocol.json',protocol)
    np.save(EXP/'whitening_mean.npy',mu);np.save(EXP/'whitening_std.npy',std)
    registry=[];selected_models={};all_runs=[]
    for name in MODELS:
        for seed in SEEDS:
            best=None;chosen_model=None
            for lr in LRS:
                model,row=fit(name,seed,lr,corpus,splits['train'],splits['validation'])
                all_runs.append(row)
                if best is None or key(row['validation'])>key(best['validation']):best,chosen_model=row,model
            filename=f'{name}_{seed}.pt'
            torch.save({'selected':chosen_model.state_dict(),'scale':best['scale']},EXP/filename)
            registry.append({**best,'checkpoint':filename,'checkpoint_sha256':sha(EXP/filename)})
            selected_models[filename]=chosen_model
            save(EXP/'training_runs.json',all_runs)
            print(f"{name}，种子{seed}：验证严格反转{key(best['validation'])[0]:.1%}，选中步数{best['selected_step']}。",flush=True)
    date_head,query_head,head_registry=fit_heads(corpus,splits['train'],splits['validation'])
    torch.save({'date':date_head,'query':query_head},EXP/'scalar_heads.pt')
    head_registry['checkpoint_sha256']=sha(EXP/'scalar_heads.pt')
    save(EXP/'head_selection.json',head_registry);save(EXP/'selection_registry.json',registry)
    print('所有读出和标量探针已冻结，开始两个独立测试条件。',flush=True)
    results=[]
    for split in ('interpolation','extrapolation'):
        subset=splits[split]
        for name in ('cosine_raw','cosine_whitened'):
            model=GeometryReadout('raw_cosine',1536).to(device=corpus['queries'].device,dtype=torch.float64)
            results.append({'split':split,'name':name,'seed':None,'cases':evaluate(model,corpus,subset,raw=name=='cosine_raw')})
        for row in registry:
            model=selected_models[row['checkpoint']]
            output=evaluate(model,corpus,subset,row['scale'])
            results.append({'split':split,'name':row['name'],'seed':row['seed'],'cases':output})
            if row['name']=='poincare_1':
                model.name='shared_euclidean'
                ablated=evaluate(model,corpus,subset,row['scale'])
                model.name='poincare_1'
                results.append({'split':split,'name':'poincare_flat_ablation','seed':row['seed'],'cases':ablated})
        for name,od,oq in (('scalar_heads',False,False),('oracle_query',False,True),('oracle_date',True,False),('both_oracles',True,True)):
            results.append({'split':split,'name':name,'seed':None,**evaluate_heads(date_head,query_head,corpus,subset,od,oq)})
    summary=[];primary={}
    all_names=('cosine_raw','cosine_whitened',*MODELS,'poincare_flat_ablation','scalar_heads','oracle_query','oracle_date','both_oracles')
    for split in ('interpolation','extrapolation'):
        for name in all_names:
            group=[r for r in results if r['split']==split and r['name']==name]
            row={'split':split,'name':name}
            for metric in ('strict','paired','ordinary','temporal_single'):
                per_case=np.mean([[c[metric] for c in r['cases']] for r in group],axis=0)
                row[metric]=bootstrap(per_case)
            if 'date_mae_days' in group[0]:row.update(date_mae_days=group[0]['date_mae_days'],query_role_accuracy=group[0]['query_role_accuracy'])
            summary.append(row)
        successes=lambda n:np.mean([[c['strict'] for c in r['cases']] for r in results if r['split']==split and r['name']==n],axis=0)
        delta=successes('poincare_1')-successes('shared_euclidean')
        rng=np.random.default_rng(20261016);samples=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
        primary[split]={'difference':float(delta.mean()),'ci975':np.quantile(samples,[.0125,.9875]).tolist(),'clusters':len(delta)}
    save(EXP/'results.json',{'protocol_sha256':sha(EXP/'protocol.json'),'selection_registry_sha256':sha(EXP/'selection_registry.json'),
                           'head_selection_sha256':sha(EXP/'head_selection.json'),'summary':summary,'primary':primary,'runs':results})
    lines=['# 冻结向量的日期反事实探索','','这是全新程序生成文本；模型在48条训练链拟合，12条验证链选模。插值、外推各24条链，日期值、物品描述和时间问法留出。两种日期配置内的查询按链聚合。','','| 条件 | 读出 | 严格日期反转 | 时间成对 | 普通单问 |','|---|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['split']} | {r['name']} | {r['strict']['mean']:.1%} | {r['paired']['mean']:.1%} | {r['ordinary']['mean']:.1%} |")
    lines+=['','主要比较为双曲减同参数欧氏；两个比较的区间各为97.5%，不使用其他模型的最高测试成绩替代主比较。']
    for split,r in primary.items():lines.append(f"- {split}：{r['difference']:.2%}，配对区间[{r['ci975'][0]:.2%}, {r['ci975'][1]:.2%}]。")
    lines+=['','标量两头探针从向量预测日期与查询类别，训练标签更强，仅用于诊断信息是否可读。oracle_query和oracle_date分别提供真查询类别或真日期，both_oracles只是定义自检；不能把oracle成绩当向量读出收益。','']
    for r in summary:
        if r['name']=='scalar_heads':lines.append(f"- {r['split']}：标量日期MAE {r['date_mae_days']:.1f}天，查询类别准确率{r['query_role_accuracy']:.1%}。")
    lines+=['','本轮仍不证明一般黎曼优势或自然文本时间记忆；新测试中的其他对照属探索性。全部选择在测试评分前冻结，旧50链未参与训练或选模。']
    (EXP/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('日期反事实探索完成。',flush=True)


if __name__=='__main__':main()
