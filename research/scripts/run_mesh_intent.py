"""同初始化、同容量的层级意图检索；验证保留普通检索后才打开测试。"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geometry_readouts import exp_origin
from prepare_mesh_intent import ROOT,DATA,EMBED,EXP,sha,save

DIM=256
MODES=('cosine','euclidean','hyperbolic')
SEEDS=(11,23,47)
LRS=(.0001,.0005,.005)
STEPS=400
EPS=1e-8
RETENTION=.02


def bounded(x):
    return x*(6/x.norm(dim=-1,keepdim=True).clamp_min(6))


def geometry_scores(q,d,mode):
    if mode=='cosine':
        return F.normalize(q,dim=-1)@F.normalize(d,dim=-1).T
    if mode=='euclidean':
        return -(q.square().sum(-1)[:,None]+d.square().sum(-1)[None,:]-2*q@d.T).clamp_min(0)
    if mode!='hyperbolic':
        raise ValueError('未知几何模式')
    # 与逐对精确Poincare距离等价，用Gram矩阵避免Q×D×维度的巨大中间张量。
    q,d=exp_origin(q,1.),exp_origin(d,1.)
    q2=q.square().sum(-1);d2=d.square().sum(-1)
    delta=(q2[:,None]+d2[None,:]-2*q@d.T).clamp_min(0)
    denominator=((1-q2[:,None])*(1-d2[None,:])).clamp_min(1e-14)
    ratio=delta/denominator
    z=ratio.clamp_min(1e-24).sqrt()
    return -4*ratio*(torch.asinh(z)/z).square()


class IntentReadout(nn.Module):
    def __init__(self,mode,basis,center):
        super().__init__()
        self.mode=mode
        # 三种几何均使用相同两组投影，允许查询与文档分布不同，参数数量完全相同。
        self.query_weight=nn.Parameter(basis.clone())
        self.document_weight=nn.Parameter(basis.clone())
        self.register_buffer('center',center.clone())

    def coordinates(self,x,query):
        weight=self.query_weight if query else self.document_weight
        return bounded((x-self.center)@weight.T)

    def forward(self,q,d):
        return geometry_scores(self.coordinates(q,True),self.coordinates(d,False),self.mode)


def positive_arrays(queries,device):
    maximum=max(len(q['positive_di']) for q in queries)
    indices=torch.zeros(len(queries),maximum,dtype=torch.long,device=device)
    valid=torch.zeros_like(indices,dtype=torch.bool)
    for i,q in enumerate(queries):
        n=len(q['positive_di'])
        indices[i,:n]=torch.tensor(q['positive_di'],device=device)
        valid[i,:n]=True
    return indices,valid


def positive_loss(scores,indices,valid):
    # 所有正确父类都获得监督，不把其他有效父类当作负例。
    logp=F.log_softmax(scores,dim=-1).gather(1,indices)
    return -((logp*valid).sum(-1)/valid.sum(-1)).mean()


def case_metrics(scores,queries,document_ids):
    if not torch.isfinite(scores).all():
        raise ValueError('分数非有限')
    indices,valid=positive_arrays(queries,scores.device)
    positive=torch.zeros_like(scores,dtype=torch.bool)
    # 避免补齐位置把第0篇文档错误标成正例。
    row=torch.arange(len(queries),device=scores.device)[:,None].expand_as(indices)
    positive[row[valid],indices[valid]]=True
    best=scores.masked_fill(~positive,-torch.inf).max(-1).values
    negative=scores.masked_fill(positive,-torch.inf)
    rank=1+(negative>=best[:,None]-EPS).sum(-1)
    recall5=positive.gather(1,scores.topk(min(5,scores.shape[1]),dim=-1).indices).sum(-1)/valid.sum(-1)
    predictions=scores.argmax(-1).tolist();ranks=rank.tolist();recalls=recall5.tolist()
    grouped={}
    for q,k,r5,prediction in zip(queries,ranks,recalls,predictions):
        grouped.setdefault(q['focus_id'],[]).append({**q,'correct':float(k==1),'rr':1/k,'recall5':r5,
                                                    'predicted_id':document_ids[prediction],'rank':k})
    cases=[]
    for focus,rows in grouped.items():
        families=sorted({q['family'] for q in rows})
        pairs=[]
        for family in families:
            pair=[q for q in rows if q['family']==family]
            if len(pair)!=2 or {q['intent'] for q in pair}!={'self','parent'}:
                raise ValueError('成对查询不完整')
            pairs.append(all(q['correct'] for q in pair))
        own=[q for q in rows if q['intent']=='self'];parents=[q for q in rows if q['intent']=='parent']
        cases.append({'id':focus,'cluster':rows[0]['cluster'],'multi_parent':len(parents[0]['positive_ids'])>1,
                      'pair':float(np.mean(pairs)),'self':float(np.mean([q['correct'] for q in own])),
                      'parent':float(np.mean([q['correct'] for q in parents])),'parent_rr':float(np.mean([q['rr'] for q in parents])),
                      'parent_recall5':float(np.mean([q['recall5'] for q in parents])),
                      'rr':float(np.mean([q['rr'] for q in rows])),
                      'parent_wrongly_self':float(np.mean([q['predicted_id']==focus for q in parents])),
                      'queries':[{k:q[k] for k in ('id','family','intent','correct','rr','recall5','predicted_id','rank')} for q in rows]})
    return cases


METRICS=('pair','self','parent','parent_rr','parent_recall5','rr','parent_wrongly_self')


def aggregate(cases):
    topics=sorted({c['cluster'] for c in cases})
    return {'macro':{m:float(np.mean([np.mean([c[m] for c in cases if c['cluster']==t]) for t in topics])) for m in METRICS},
            'micro':{m:float(np.mean([c[m] for c in cases])) for m in METRICS},'concepts':len(cases),'branches':len(topics)}


def load():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    runtime=json.loads((EMBED/'runtime_manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['input_sha256'].items():
        if sha(DATA/name)!=h:
            raise ValueError('冻结输入发生变化')
    if runtime['request_model']!='Qwen3-Embedding-8B' or runtime['dimensions']!=1536:
        raise ValueError('编码模型或维度不一致')
    for kind in ('documents','queries'):
        if runtime['input_sha256'][kind]!=manifest['input_sha256'][kind+'.jsonl']:
            raise ValueError('编码输入哈希不符')
    annotations=json.loads((DATA/'annotations.json').read_text(encoding='utf-8'))
    document_ids=json.loads((EMBED/'document_ids.json').read_text(encoding='utf-8'))
    query_ids=json.loads((EMBED/'query_ids.json').read_text(encoding='utf-8'))
    didx={v:i for i,v in enumerate(document_ids)};qidx={v:i for i,v in enumerate(query_ids)}
    for q in annotations:
        q['qi']=qidx[q['id']];q['positive_di']=[didx[p] for p in q['positive_ids']]
    arrays={k:np.load(EMBED/(k+'.npy')) for k in ('documents','queries')}
    train_docs=sorted({i for q in annotations if q['split']=='train' for i in q['positive_di']})
    mu=arrays['documents'][train_docs].mean(0);std=np.maximum(arrays['documents'][train_docs].std(0),1e-3)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus={}
    for k,a in arrays.items():
        if not np.isfinite(a).all() or np.any(np.linalg.norm(a,axis=1)==0):
            raise ValueError('向量包含非有限值或零向量')
        for prefix,v in (('raw_',a),('',(a-mu)/std)):
            corpus[prefix+k]=torch.tensor(v/np.linalg.norm(v,axis=1,keepdims=True),dtype=torch.float64,device=device)
    fit_docs=corpus['documents'][train_docs]
    center=fit_docs.mean(0)
    _,_,basis=torch.linalg.svd(fit_docs-center,full_matrices=False)
    if len(basis)<DIM:
        raise ValueError('训练正例文档不足以估计PCA维度')
    return annotations,corpus,document_ids,center,basis[:DIM],mu,std,train_docs


def evaluate(model,queries,corpus,document_ids,allow_test=False):
    if not allow_test and any(q['split']=='test' for q in queries):
        raise ValueError('选择冻结前禁止测试评分')
    with torch.no_grad():
        prefix='raw_' if model is None else ''
        q=corpus[prefix+'queries'][[r['qi'] for r in queries]];d=corpus[prefix+'documents']
        scores=q@d.T if model is None else model(q,d)
        return case_metrics(scores,queries,document_ids)


def selection_key(cases,minimum_self):
    metrics=aggregate(cases)['macro']
    if metrics['self']<minimum_self-1e-12:
        return None
    return (metrics['pair'],metrics['parent'],metrics['rr'])


def fit(mode,seed,lr,basis,center,train,val,corpus,document_ids,minimum_self):
    model=IntentReadout(mode,basis,center)
    q=corpus['queries'][[r['qi'] for r in train]];d=corpus['documents']
    positives,valid=positive_arrays(train,q.device)
    calibration_rng=np.random.default_rng(20261016)
    sample=torch.tensor(calibration_rng.choice(len(q),min(128,len(q)),replace=False),device=q.device)
    with torch.no_grad():
        initial=model(q[sample],d)
        scale=float((initial-initial.mean(-1,keepdim=True)).square().mean().sqrt().clamp_min(.01))
    baseline=evaluate(model,val,corpus,document_ids)
    key=selection_key(baseline,minimum_self)
    best={'step':0,'key':key,'validation':baseline,'state':copy.deepcopy(model.state_dict())} if key is not None else None
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    rng=np.random.default_rng(seed);history=[]
    for step in range(1,STEPS+1):
        ix=torch.tensor(rng.choice(len(train),min(64,len(train)),replace=False),device=q.device)
        opt.zero_grad(set_to_none=True)
        loss=positive_loss(model(q[ix],d)/scale,positives[ix],valid[ix])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%50==0 or step==STEPS:
            current=evaluate(model,val,corpus,document_ids)
            history.append({'step':step,'train_loss':float(loss.detach()),'validation':aggregate(current)})
            key=selection_key(current,minimum_self)
            if key is not None and (best is None or key>best['key']):
                best={'step':step,'key':key,'validation':current,'state':copy.deepcopy(model.state_dict())}
    return best,{'mode':mode,'seed':seed,'lr':lr,'scale':scale,'initial_validation':aggregate(baseline),'history':history,
                 'selected_step':None if best is None else best['step']}


def paired_interval(runs,left,right):
    topics=sorted({c['cluster'] for r in runs for c in r['cases']})
    means={}
    for name in (left,right):
        group=[r for r in runs if r['name']==name]
        means[name]=np.array([np.mean([np.mean([c['pair'] for c in r['cases'] if c['cluster']==t]) for r in group]) for t in topics])
    delta=means[left]-means[right];rng=np.random.default_rng(20261016)
    draws=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(1)
    return {'difference':float(delta.mean()),'ci975':np.quantile(draws,[.0125,.9875]).tolist(),'branches':len(topics)}


def execute():
    lock=json.loads((EXP/'encoding_lock.json').read_text(encoding='utf-8'))
    if sha(DATA/'manifest.json')!=lock['manifest_sha256']:
        raise ValueError('冻结数据清单不符')
    for name,h in lock['code_sha256'].items():
        if sha(ROOT/'scripts'/name)!=h:
            raise ValueError('冻结代码不符')
    if (EXP/'selection_registry.json').exists() or (EXP/'protocol.json').exists():
        raise RuntimeError('本轮已有训练或选择记录，禁止自动覆盖')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    records,corpus,document_ids,center,basis,mu,std,fit_indices=load()
    splits={s:[q for q in records if q['split']==s] for s in ('train','validation','test')}
    raw_val=evaluate(None,splits['validation'],corpus,document_ids)
    minimum_self=aggregate(raw_val)['macro']['self']-RETENTION
    torch.save({'center':center,'basis':basis,'whitening_mean':mu,'whitening_std':std,'fit_document_indices':fit_indices},EXP/'pca_initialization.pt')
    protocol={'scope':'真实MeSH定义与关系上的模板意图机制检验，非自然用户检索或HypRAG整模型复现。',
              'modes':MODES,'seeds':SEEDS,'learning_rates':LRS,'steps':STEPS,'dimension':DIM,'dtype':'float64','device':str(corpus['documents'].device),
              'architecture':f'相同查询/文档两组1536→{DIM}投影，共{2*1536*DIM}个可训练参数；同一PCA初始化、同一半径6约束，固定K=-1。',
              'supervision':'同一训练查询与全部有效正例；全5000文档softmax，所有正确父类的平均负对数概率。无本体边、焦点ID或真意图作为模型输入。',
              'selection':'只在验证集满足self宏R@1不低于原始cosine减2个百分点时，依次最大化pair、parent、RR；包括第0步。所有模型和种子通过才开测试。',
              'minimum_validation_self':minimum_self,'validation_raw':aggregate(raw_val),'pca_fit_documents':len(fit_indices),
              'primary':'测试成对意图宏R@1；双曲减欧氏、双曲减学习cosine，种子先平均，按规范分支配对bootstrap20000次，分别97.5%区间。',
              'practical_success':'两个主要差值均至少+2个百分点且区间下界>0，同时测试self宏R@1不低于原始cosine减2个百分点。区间宽则不确定，不能宣称无收益。',
              'cross_evaluation':'欧氏与双曲选中检查点固定切空间坐标，交叉替换评分距离。',
              'manifest_sha256':sha(DATA/'manifest.json'),'pca_initialization_sha256':sha(EXP/'pca_initialization.pt'),
              'vector_sha256':{n:sha(EMBED/n) for n in ('documents.npy','queries.npy','runtime_manifest.json')},'code_sha256':lock['code_sha256']}
    save(EXP/'protocol.json',protocol)
    registry=[];models={};training=[];failures=[]
    for mode in MODES:
        for seed in SEEDS:
            selected=None
            for lr in LRS:
                candidate,run=fit(mode,seed,lr,basis,center,splits['train'],splits['validation'],corpus,document_ids,minimum_self)
                training.append(run);save(EXP/'training_runs.json',training)
                if candidate is not None and (selected is None or candidate['key']>selected['key']):
                    selected={**candidate,'lr':lr,'scale':run['scale']}
            if selected is None:
                failures.append({'mode':mode,'seed':seed,'reason':'没有检查点通过普通定义检索保留门槛'})
                print(f'{mode}，种子{seed}：验证保留门槛未通过。',flush=True)
                continue
            model=IntentReadout(mode,basis,center);model.load_state_dict(selected['state'])
            filename=f'{mode}_{seed}.pt'
            torch.save(model.state_dict(),EXP/filename)
            registry.append({'mode':mode,'seed':seed,'lr':selected['lr'],'step':selected['step'],'scale':selected['scale'],
                             'validation':aggregate(selected['validation']),'checkpoint':filename,'sha256':sha(EXP/filename),
                             'parameters':sum(p.numel() for p in model.parameters())})
            models[filename]=model
            print(f"{mode}，种子{seed}：验证成对{selected['key'][0]:.1%}，self={aggregate(selected['validation'])['macro']['self']:.1%}，选中步数{selected['step']}。",flush=True)
    save(EXP/'selection_registry.json',{'selected':registry,'failures':failures})
    if failures:
        save(EXP/'results.json',{'stage':'验证门槛未通过，测试保持封存','protocol_sha256':sha(EXP/'protocol.json'),'failures':failures,'selected':registry,'test_scored':False})
        (EXP/'report.md').write_text('# 层级意图检验\n\n部分模型未通过预先冻结的普通定义检索保留门槛，未进行测试评分。详见results.json及training_runs.json。\n',encoding='utf-8')
        return False
    if len({r['parameters'] for r in registry})!=1:
        raise ValueError('三个模型参数数量不一致')
    print('全部模型通过验证门槛且选择冻结，开始唯一一次测试评分。',flush=True)
    test=splits['test'];runs=[]
    for name,model in (('raw_cosine',None),('pca_cosine',IntentReadout('cosine',basis,center))):
        runs.append({'name':name,'seed':None,'cases':evaluate(model,test,corpus,document_ids,True)})
    changes=[]
    for r in registry:
        model=models[r['checkpoint']]
        original=evaluate(model,test,corpus,document_ids,True)
        runs.append({'name':r['mode'],'seed':r['seed'],'cases':original})
        if r['mode'] in ('euclidean','hyperbolic'):
            model.mode='euclidean' if r['mode']=='hyperbolic' else 'hyperbolic'
            swapped=evaluate(model,test,corpus,document_ids,True)
            alias='hyper_trained_flat' if r['mode']=='hyperbolic' else 'flat_trained_hyper'
            runs.append({'name':alias,'seed':r['seed'],'cases':swapped})
            before={q['id']:q['predicted_id'] for c in original for q in c['queries']}
            after={q['id']:q['predicted_id'] for c in swapped for q in c['queries']}
            changes.append({'mode':r['mode'],'seed':r['seed'],'changed':sum(before[q]!=after[q] for q in before),'queries':len(before)})
            model.mode=r['mode']
    summary=[]
    for name in sorted({r['name'] for r in runs}):
        group=[r for r in runs if r['name']==name]
        summary.append({'name':name,'seeds':len(group),
                        'macro':{m:float(np.mean([aggregate(r['cases'])['macro'][m] for r in group])) for m in METRICS},
                        'micro':{m:float(np.mean([aggregate(r['cases'])['micro'][m] for r in group])) for m in METRICS},
                        'strata':{str(multi):{m:float(np.mean([np.mean([c[m] for c in r['cases'] if c['multi_parent']==multi]) for r in group])) for m in METRICS} for multi in (False,True)}})
    primary={'hyper_minus_euclidean':paired_interval(runs,'hyperbolic','euclidean'),
             'hyper_minus_cosine':paired_interval(runs,'hyperbolic','cosine')}
    by_name={r['name']:r for r in summary}
    success=all(r['difference']>=.02 and r['ci975'][0]>0 for r in primary.values()) and by_name['hyperbolic']['macro']['self']>=by_name['raw_cosine']['macro']['self']-RETENTION
    checks={'checkpoint_hashes':all(sha(EXP/r['checkpoint'])==r['sha256'] for r in registry),
            'pca_hash':sha(EXP/'pca_initialization.pt')==protocol['pca_initialization_sha256']}
    if not all(checks.values()):
        raise ValueError('完成后的检查点核验失败')
    save(EXP/'results.json',{'stage':'已完成','test_scored':True,'protocol_sha256':sha(EXP/'protocol.json'),
                            'registry_sha256':sha(EXP/'selection_registry.json'),'summary':summary,'primary':primary,
                            'practical_success':success,'cross_prediction_changes':changes,'checks':checks,'runs':runs})
    lines=['# MeSH层级意图检索结果','','真实定义与全部直接父关系上的模板机制评测。5000文档，测试200个焦点概念；三个模型均为256维、相同两组PCA初始化投影。所有验证保留门槛通过后，冻结选择并只打开一次测试。','','| 读出 | 成对意图宏正确率 | 定义宏正确率 | 父类宏正确率 | 父类Recall@5宏 |','|---|---:|---:|---:|---:|']
    for r in summary:
        m=r['macro'];lines.append(f"| {r['name']} | {m['pair']:.1%} | {m['self']:.1%} | {m['parent']:.1%} | {m['parent_recall5']:.1%} |")
    lines+=['','两个主要配对比较各报告97.5%区间：']
    for name,r in primary.items():
        lines.append(f"- {name}：{r['difference']:.2%}，区间[{r['ci975'][0]:.2%}, {r['ci975'][1]:.2%}]，{r['branches']}个规范分支。")
    lines+=['',f'达到预先指定的曲率实用收益标准：{success}。','模板查询与额外关系监督不能直接外推到自然用户检索、无监督距离替换或长期记忆。测试已经打开，不再用于调参。','']
    (EXP/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('层级意图实验完成。',flush=True)
    return True


if __name__=='__main__':
    execute()
