"""冻结向量上的真实层级候选检索；区分训练几何与评分距离。"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geometry_readouts import GeometryReadout,tangent_distance_squared
from prepare_pubmed_geometry import ROOT,DATA,EMBED,EXP,save,sha

MODELS=('shared_cosine','shared_euclidean','poincare_1','nonlinear_euclidean')
SEEDS=(11,23,47,59,83)
LRS=(.0005,.005)
STEPS=800
DIM=8
EPS=1e-8


class Readout(GeometryReadout):
    def __init__(self,name):
        super().__init__('shared_euclidean' if name=='nonlinear_euclidean' else name,1536,DIM)
        if name=='nonlinear_euclidean':
            # 无额外参数的非线性对照；所有学习模型的参数数量完全一致。
            self.map=nn.Sequential(self.map,nn.Tanh())
        self.name=name


def ranks(scores,target):
    actual=scores.gather(1,target[:,None])
    other=scores.clone()
    other.scatter_(1,target[:,None],-torch.inf)
    # 平手保守处理；不让候选位置决定正确率。
    rank=1+(other>=actual-EPS).sum(-1)
    return rank,other.argmax(-1)


def metrics(scores,queries,targets):
    rank,_=ranks(scores,targets)
    prediction=scores.argmax(-1).tolist()
    rows=[]
    for i,(q,r) in enumerate(zip(queries,rank.tolist())):
        rows.append({'id':q['id'],'cluster':q['cluster'],'rank':r,'r1':float(r==1),'rr':1/r,
                     'top_id':q['candidates'][prediction[i]]['id'],
                     'top_type':q['candidates'][prediction[i]]['type']})
    return rows


def aggregate(rows):
    topics=sorted({r['cluster'] for r in rows})
    by_topic=[{m:float(np.mean([r[m] for r in rows if r['cluster']==t])) for m in ('r1','rr')} for t in topics]
    return {'r1':float(np.mean([r['r1'] for r in rows])),'mrr':float(np.mean([r['rr'] for r in rows])),
            'macro_r1':float(np.mean([r['r1'] for r in by_topic])),
            'macro_mrr':float(np.mean([r['rr'] for r in by_topic])),'topics':len(topics),'queries':len(rows)}


def load():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    runtime=json.loads((EMBED/'runtime_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['input_sha256'].items():
        if sha(DATA/name)!=expected:
            raise ValueError('冻结数据发生变化')
    if runtime['request_model']!='Qwen3-Embedding-8B' or runtime['dimensions']!=1536:
        raise ValueError('向量来源不符')
    for kind in ('documents','queries'):
        if runtime['input_sha256'][kind]!=manifest['input_sha256'][kind+'.jsonl']:
            raise ValueError('编码输入哈希不符')
    queries=json.loads((DATA/'annotations.json').read_text(encoding='utf-8'))
    arrays={k:np.load(EMBED/(k+'.npy')) for k in ('documents','queries')}
    index={k:{v:i for i,v in enumerate(json.loads((EMBED/(('document' if k=='documents' else 'query')+'_ids.json')).read_text(encoding='utf-8')))} for k in arrays}
    train=[index['documents'][q['target_id']] for q in queries if q['split']=='train']
    mu=arrays['documents'][train].mean(0)
    std=np.maximum(arrays['documents'][train].std(0),1e-3)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vectors={}
    for name,a in arrays.items():
        if not np.isfinite(a).all() or np.any(np.linalg.norm(a,axis=1)==0):
            raise ValueError('向量包含非有限值或零向量')
        for prefix,values in (('raw_',a),('',(a-mu)/std)):
            normalized=values/np.linalg.norm(values,axis=1,keepdims=True)
            vectors[prefix+name]=torch.tensor(normalized,device=device,dtype=torch.float64)
    for q in queries:
        q['qi']=index['queries'][q['id']]
        q['di']=[index['documents'][d['id']] for d in q['candidates']]
        q['target_di']=index['documents'][q['target_id']]
    return queries,vectors,mu,std,index


def assemble(queries,vectors,raw=False):
    prefix='raw_' if raw else ''
    q=vectors[prefix+'queries'][[r['qi'] for r in queries]]
    indices=torch.tensor([r['di'] for r in queries],device=q.device,dtype=torch.long)
    d=vectors[prefix+'documents'][indices]
    y=torch.tensor([r['target_index'] for r in queries],device=q.device)
    return q,d,y


def evaluate(model,queries,vectors,scale=1.,raw=False):
    with torch.no_grad():
        q,d,y=assemble(queries,vectors,raw)
        scores=model(q,d)/scale
        if not torch.isfinite(scores).all():
            raise ValueError('检索分数非有限')
        return metrics(scores,queries,y)


def fit(name,seed,lr,train,val,vectors):
    torch.manual_seed(seed)
    model=Readout(name).to(device=vectors['queries'].device,dtype=torch.float64)
    q,d,y=assemble(train,vectors)
    with torch.no_grad():
        scores=model(q,d)
        scale=float((scores-scores.mean(-1,keepdim=True)).square().mean().sqrt().clamp_min(.01))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best=aggregate(evaluate(model,val,vectors,scale))
    best_state=copy.deepcopy(model.state_dict());best_step=0
    key=lambda r:(r['macro_r1'],r['macro_mrr'])
    history=[];rng=np.random.default_rng(seed)
    for step in range(1,STEPS+1):
        ix=torch.tensor(rng.choice(len(train),64,replace=False),device=q.device)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(q[ix],d[ix])/scale,y[ix])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%100==0:
            current=aggregate(evaluate(model,val,vectors,scale))
            history.append({'step':step,'validation':current,'train_loss':float(loss.detach())})
            if key(current)>key(best):
                best,best_state,best_step=current,copy.deepcopy(model.state_dict()),step
    model.load_state_dict(best_state)
    return model,{'name':name,'seed':seed,'lr':lr,'selected_step':best_step,'scale':scale,'validation':best,
                  'history':history,'parameters':sum(p.numel() for p in model.parameters())}


def full_library(model,queries,vectors,index,raw=False):
    prefix='raw_' if raw else ''
    with torch.no_grad():
        q=vectors[prefix+'queries'][[r['qi'] for r in queries]]
        d=vectors[prefix+'documents']
        if model.name=='raw_cosine':
            scores=q@d.T
        else:
            qv,dv=model.coordinates(q),model.coordinates(d)
            if model.name=='shared_cosine':
                scores=F.normalize(qv,dim=-1)@F.normalize(dv,dim=-1).T
            else:
                scores=-tangent_distance_squared(qv[:,None,:],dv[None,:,:],model.curvature())
        if not torch.isfinite(scores).all():
            raise ValueError('全库评分非有限')
        targets=torch.tensor([r['target_di'] for r in queries],device=q.device)
        rank,_=ranks(scores,targets)
        predicted=scores.argmax(-1).tolist()
        ids={i:v for v,i in index['documents'].items()}
        return [{'id':r['id'],'cluster':r['cluster'],'rank':k,'r1':float(k==1),'rr':1/k,'top_id':ids[p]}
                for r,k,p in zip(queries,rank.tolist(),predicted)]


def paired_interval(runs,left,right,metric='r1'):
    topics=sorted({c['cluster'] for r in runs for c in r['local']})
    values={}
    for name in (left,right):
        selected=[r for r in runs if r['name']==name]
        values[name]=np.array([np.mean([np.mean([c[metric] for c in r['local'] if c['cluster']==t]) for r in selected]) for t in topics])
    delta=values[left]-values[right]
    rng=np.random.default_rng(20261016)
    boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(1)
    return {'difference':float(delta.mean()),'ci975':np.quantile(boot,[.0125,.9875]).tolist(),'topics':len(topics)}


def main():
    if (EXP/'selection_registry.json').exists():
        raise RuntimeError('本轮选择已经冻结，禁止重复选模')
    lock=json.loads((EXP/'encoding_lock.json').read_text(encoding='utf-8'))
    if sha(DATA/'manifest.json')!=lock['manifest_sha256']:
        raise ValueError('编码前数据清单发生变化')
    for name,expected in lock['code_sha256'].items():
        if sha(ROOT/'scripts'/name)!=expected:
            raise ValueError('冻结的模型或评估代码发生变化')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    queries,vectors,mu,std,index=load()
    splits={s:[q for q in queries if q['split']==s] for s in ('train','validation','test')}
    save(EXP/'protocol.json',{'stage':'测试评分前','primary':'两个比较：双曲训练/双曲评分减欧氏训练/欧氏评分；双曲训练/欧氏评分减欧氏训练/欧氏评分。主指标按MeSH规范分支宏平均R@1。',
                             'interval':'按分支配对bootstrap20000次；两个主要比较分别97.5%区间。种子先平均，不作为独立样本。',
                             'models':MODELS,'seeds':SEEDS,'lrs':LRS,'steps':STEPS,'dimension':DIM,'dtype':'float64',
                             'device':str(vectors['queries'].device),'selection':'只用验证分支宏平均R@1，MRR打破平手；相同预算选学习率和检查点。',
                             'cross_evaluation':'双曲训练切平坦评分、欧氏训练切双曲评分；固定投影后的切空间坐标、检查点、输入标准化。交叉评分沿用按各模型原生距离验证选中的检查点。',
                             'supervision':'各模型只接收相同训练标题、候选摘要和对应文章标签；无MeSH结构输入、无测试标签训练。',
                             'source':'公开真实文章与专家MeSH；独立构造已知文章检索诊断，不是HypRAG全模型复现。',
                             'manifest_sha256':sha(DATA/'manifest.json'),
                             'vector_sha256':{k:sha(EMBED/k) for k in ('documents.npy','queries.npy','runtime_manifest.json')},
                             'code_sha256':lock['code_sha256']})
    np.save(EXP/'whitening_mean.npy',mu);np.save(EXP/'whitening_std.npy',std)
    registry=[];models={};training=[]
    for name in MODELS:
        for seed in SEEDS:
            best=None;selected=None
            for lr in LRS:
                model,row=fit(name,seed,lr,splits['train'],splits['validation'],vectors)
                training.append(row)
                if best is None or (row['validation']['macro_r1'],row['validation']['macro_mrr'])>(best['validation']['macro_r1'],best['validation']['macro_mrr']):
                    best,selected=row,model
            checkpoint=f'{name}_{seed}.pt'
            torch.save({'state_dict':selected.state_dict(),'scale':best['scale']},EXP/checkpoint)
            registry.append({**best,'checkpoint':checkpoint,'checkpoint_sha256':sha(EXP/checkpoint)})
            models[checkpoint]=selected
            save(EXP/'training_runs.json',training)
            print(f"{name}，种子{seed}：验证宏平均R@1={best['validation']['macro_r1']:.1%}，选中步数{best['selected_step']}。",flush=True)
    if len({r['parameters'] for r in registry})!=1:
        raise ValueError('学习模型参数数量不一致')
    save(EXP/'selection_registry.json',registry)
    print('全部检查点和选择已冻结，开始唯一一次测试评分。',flush=True)
    runs=[];test=splits['test']
    for name,raw in (('cosine_raw',True),('cosine_whitened',False)):
        model=Readout('raw_cosine').to(device=vectors['queries'].device,dtype=torch.float64)
        runs.append({'name':name,'seed':None,'local':evaluate(model,test,vectors,raw=raw),
                     'global':full_library(model,test,vectors,index,raw)})
    changes=[]
    for row in registry:
        model=models[row['checkpoint']]
        local=evaluate(model,test,vectors,row['scale'])
        runs.append({'name':row['name'],'seed':row['seed'],'local':local,'global':full_library(model,test,vectors,index)})
        if row['name'] in ('shared_euclidean','poincare_1'):
            model.name='shared_euclidean' if row['name']=='poincare_1' else 'poincare_1'
            swapped=evaluate(model,test,vectors,row['scale'])
            alias='hyper_trained_flat' if row['name']=='poincare_1' else 'flat_trained_hyper'
            runs.append({'name':alias,'seed':row['seed'],'local':swapped,'global':full_library(model,test,vectors,index)})
            changes.append({'trained':row['name'],'seed':row['seed'],'top1_changes':sum(a['top_id']!=b['top_id'] for a,b in zip(local,swapped)),'queries':len(test)})
            model.name=row['name']
    primary={'complete_hyper_minus_flat':paired_interval(runs,'poincare_1','shared_euclidean'),
             'hyper_training_under_flat_scoring':paired_interval(runs,'hyper_trained_flat','shared_euclidean')}
    summary=[]
    for name in sorted({r['name'] for r in runs}):
        group=[r for r in runs if r['name']==name]
        result={'name':name,'seeds':len(group)}
        for condition in ('local','global'):
            aggregates=[aggregate(r[condition]) for r in group]
            result[condition]={m:float(np.mean([r[m] for r in aggregates])) for m in ('r1','mrr','macro_r1','macro_mrr')}
            result[condition].update(queries=aggregates[0]['queries'],topics=aggregates[0]['topics'])
        result['top_type_counts']={t:float(np.mean([sum(c.get('top_type')==t for c in r['local']) for r in group])) for t in ('target','sibling','parent','grandparent','random')}
        summary.append(result)
    verification={'protocol_sha256':sha(EXP/'protocol.json'),'selection_registry_sha256':sha(EXP/'selection_registry.json'),
                  'all_checkpoint_hashes_match':all(sha(EXP/r['checkpoint'])==r['checkpoint_sha256'] for r in registry)}
    save(EXP/'results.json',{'verification':verification,'primary':primary,'summary':summary,'cross_prediction_changes':changes,'runs':runs})
    lines=['# PubMed真实语料上的几何读出实验','','这是独立构造的已知文章检索诊断；标题匹配对应摘要。主评测含10个固定候选，次评测在全部5000篇摘要中检索。MeSH规范分支留出，模型只用相同训练相关性标签。','','| 读出 | 困难候选宏R@1 | 困难候选MRR | 全库宏R@1 |','|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['name']} | {r['local']['macro_r1']:.1%} | {r['local']['mrr']:.3f} | {r['global']['macro_r1']:.1%} |")
    lines+=['','两个主要配对比较，分别97.5%区间：']
    for name,r in primary.items():
        lines.append(f"- {name}：{r['difference']:.2%}，区间[{r['ci975'][0]:.2%}, {r['ci975'][1]:.2%}]，{r['topics']}个测试分支。")
    lines+=['','向量维度1536，投影维度8；各学习模型参数数量相同。非线性欧氏为同一个线性投影后接无参数Tanh。全部选择在测试评分前冻结。','模型之间仍有几何导致的优化差异；收益不等同于证明一般黎曼优势。不能将此诊断直接外推到长期记忆。','']
    (EXP/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('真实语料几何读出实验完成，结果与审计已保存。',flush=True)


if __name__=='__main__':
    main()
