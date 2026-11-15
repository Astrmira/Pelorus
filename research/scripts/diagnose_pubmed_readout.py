"""仅用训练与验证查询，诊断维度压缩、训练负例及验证选择的影响。"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy
import json
from pathlib import Path
import traceback
from datetime import datetime,UTC
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_pubmed_geometry import ROOT,DATA,EMBED,sha,save

EXP=ROOT/'artifacts/experiments/pubmed_readout_diagnostics_v1'
OLD=ROOT/'artifacts/experiments/pubmed_geometry_v1'
DIMS=(8,32,128,256)
SEEDS=(11,23,47)
LRS=(.0005,.005)
STEPS=400
CONDITIONS=(('random','local'),('random','full'),('pca','full'))
EPS=1e-6


def restrict_queries(rows):
    """测试查询在送入设备之前排除，任何未知划分直接报错。"""
    if any(q['split'] not in ('train','validation','test') for q in rows):
        raise ValueError('存在未知数据划分')
    return [q for q in rows if q['split'] in ('train','validation')]


def guard_queries(rows):
    if any(q['split'] not in ('train','validation') for q in rows):
        raise ValueError('诊断禁止评分测试查询')


def load():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    old_protocol=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    for name,h in manifest['input_sha256'].items():
        if sha(DATA/name)!=h:
            raise ValueError('原始冻结输入发生变化')
    for name,h in old_protocol['vector_sha256'].items():
        if sha(EMBED/name)!=h:
            raise ValueError('原始冻结向量发生变化')
    records=restrict_queries(json.loads((DATA/'annotations.json').read_text(encoding='utf-8')))
    dids=json.loads((EMBED/'document_ids.json').read_text(encoding='utf-8'))
    qids=json.loads((EMBED/'query_ids.json').read_text(encoding='utf-8'))
    di={v:i for i,v in enumerate(dids)};qi={v:i for i,v in enumerate(qids)}
    documents=np.load(EMBED/'documents.npy')
    # 只提取300个训练和验证查询向量，不将测试查询复制到设备。
    queries=np.asarray(np.load(EMBED/'queries.npy',mmap_mode='r')[[qi[q['id']] for q in records]])
    for i,q in enumerate(records):
        q['qi']=i;q['di']=[di[d['id']] for d in q['candidates']];q['target_di']=di[q['target_id']]
    train_target=[q['target_di'] for q in records if q['split']=='train']
    mu=documents[train_target].mean(0);std=np.maximum(documents[train_target].std(0),1e-3)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    vectors={}
    for name,a in (('documents',documents),('queries',queries)):
        for prefix,v in (('raw_',a),('',(a-mu)/std)):
            if not np.isfinite(v).all() or np.any(np.linalg.norm(v,axis=1)==0):
                raise ValueError('输入存在非有限值或零向量')
            vectors[prefix+name]=torch.tensor(v/np.linalg.norm(v,axis=1,keepdims=True),device=device,dtype=torch.float32)
    return records,vectors,{'documents':di,'queries':{q['id']:q['qi'] for q in records}}


def pca_fit(documents,indices):
    fit=documents[indices].to(torch.float64)
    center=fit.mean(0)
    # 全矩阵SVD，不按验证集挑成分；仅拟合训练候选文档。
    _,_,basis=torch.linalg.svd(fit-center,full_matrices=False)
    return center.to(documents.dtype),basis.to(documents.dtype)


def random_basis(dim,seed,device,dtype):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    matrix=torch.randn(1536,dim,generator=generator,dtype=torch.float64)
    q,_=torch.linalg.qr(matrix,mode='reduced')
    return q.T.to(device=device,dtype=dtype)


class CosineHead(nn.Module):
    def __init__(self,basis,center):
        super().__init__()
        self.weight=nn.Parameter(basis.clone())
        self.register_buffer('center',center.clone())

    def forward(self,x):
        return F.normalize((x-self.center)@self.weight.T,dim=-1)


def score_matrix(model,q,documents):
    return model(q)@model(documents).T


def row_metrics(scores,queries,full=True):
    guard_queries(queries)
    if not torch.isfinite(scores).all():
        raise ValueError('分数非有限')
    targets=[q['target_di'] if full else q['target_index'] for q in queries]
    y=torch.tensor(targets,device=scores.device)
    target=scores.gather(1,y[:,None])
    other=scores.clone();other.scatter_(1,y[:,None],-torch.inf)
    rank=(1+(other>=target-EPS).sum(-1)).tolist()
    return [{'id':q['id'],'cluster':q['cluster'],'r1':float(r==1),'rr':1/r,'rank':r} for q,r in zip(queries,rank)]


def aggregate(rows):
    topics=sorted({r['cluster'] for r in rows})
    return {'r1':float(np.mean([r['r1'] for r in rows])),'mrr':float(np.mean([r['rr'] for r in rows])),
            'macro_r1':float(np.mean([np.mean([r['r1'] for r in rows if r['cluster']==t]) for t in topics])),
            'macro_mrr':float(np.mean([np.mean([r['rr'] for r in rows if r['cluster']==t]) for t in topics])),
            'queries':len(rows),'topics':len(topics)}


def evaluate(model,queries,vectors,raw=False):
    guard_queries(queries)
    prefix='raw_' if raw else ''
    with torch.no_grad():
        q=vectors[prefix+'queries'][[r['qi'] for r in queries]]
        d=vectors[prefix+'documents']
        scores=q@d.T if model is None else score_matrix(model,q,d)
        index=torch.tensor([r['di'] for r in queries],device=q.device)
        local=scores.gather(1,index)
        return {'full':row_metrics(scores,queries),'local':row_metrics(local,queries,False)}


def key(metrics,condition='full'):
    a=aggregate(metrics[condition])
    return (a['macro_r1'],a['macro_mrr'])


def fit(basis,center,objective,seed,lr,train,val,vectors):
    guard_queries(train);guard_queries(val)
    model=CosineHead(basis,center)
    q=vectors['queries'][[r['qi'] for r in train]];documents=vectors['documents']
    ci=torch.tensor([r['di'] for r in train],device=q.device)
    yt=torch.tensor([r['target_di'] for r in train],device=q.device)
    yl=torch.tensor([r['target_index'] for r in train],device=q.device)
    with torch.no_grad():
        calibration=score_matrix(model,q,documents)
        # 两个训练目标使用同一个全库初始标度，避免温度成为额外变量。
        scale=float((calibration-calibration.mean(-1,keepdim=True)).square().mean().sqrt().clamp_min(.01))
    initial=evaluate(model,val,vectors)
    selected={c:{'step':0,'validation':initial,'state':copy.deepcopy(model.state_dict())} for c in ('full','local')}
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    rng=np.random.default_rng(seed);history=[]
    for step in range(1,STEPS+1):
        ix=torch.tensor(rng.choice(len(train),min(64,len(train)),replace=False),device=q.device)
        opt.zero_grad(set_to_none=True)
        if objective=='full':
            score=score_matrix(model,q[ix],documents);target=yt[ix]
        else:
            score=(model(q[ix])[:,None,:]*model(documents[ci[ix]])).sum(-1);target=yl[ix]
        loss=F.cross_entropy(score/scale,target)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%50==0 or step==STEPS:
            current=evaluate(model,val,vectors)
            history.append({'step':step,'train_loss':float(loss.detach()),
                            'validation':{c:aggregate(current[c]) for c in ('full','local')}})
            for c in ('full','local'):
                if key(current,c)>key(selected[c]['validation'],c):
                    selected[c]={'step':step,'validation':current,'state':copy.deepcopy(model.state_dict())}
    return selected,{'seed':seed,'lr':lr,'scale':scale,'history':history,
                     'initial_validation':{c:aggregate(initial[c]) for c in ('full','local')}}


def record(name,dim,seed,model,train,val,vectors,selection=None,raw=False):
    rows={'name':name,'dimension':dim,'seed':seed,'selection':selection}
    for split,queries in (('train',train),('validation',val)):
        output=evaluate(model,queries,vectors,raw)
        rows[split]={'cases':output,'metrics':{c:aggregate(output[c]) for c in ('full','local')}}
    return rows


def summarize(rows):
    summary=[]
    for name,dim in sorted({(r['name'],r['dimension']) for r in rows}):
        group=[r for r in rows if r['name']==name and r['dimension']==dim]
        row={'name':name,'dimension':dim,'seeds':len(group)}
        for split in ('train','validation'):
            row[split]={condition:{m:float(np.mean([r[split]['metrics'][condition][m] for r in group]))
                                  for m in ('r1','mrr','macro_r1','macro_mrr')} for condition in ('full','local')}
        if group[0]['selection']:
            row['selected_steps']=[r['selection']['step'] for r in group]
        summary.append(row)
    return summary


def old_checkpoint_diagnostic(train,val,vectors,index):
    # 只读取旧验证选择清单，不读取旧results.json中的已打开测试成绩。
    from run_pubmed_geometry import Readout,full_library,evaluate as old_eval,aggregate as old_aggregate
    registry=json.loads((OLD/'selection_registry.json').read_text(encoding='utf-8'))
    v={k:t.to(torch.float64) for k,t in vectors.items()}
    rows=[]
    for selected in registry:
        if selected['seed'] not in SEEDS or selected['name'] not in ('shared_cosine','shared_euclidean','poincare_1'):
            continue
        if sha(OLD/selected['checkpoint'])!=selected['checkpoint_sha256']:
            raise ValueError('旧检查点被修改')
        model=Readout(selected['name']).to(device=v['queries'].device,dtype=torch.float64)
        model.load_state_dict(torch.load(OLD/selected['checkpoint'],map_location=v['queries'].device,weights_only=True)['state_dict'])
        row={'name':'old_'+selected['name'],'dimension':8,'seed':selected['seed'],'selection':None}
        for split,queries in (('train',train),('validation',val)):
            guard_queries(queries)
            local=old_eval(model,queries,v,selected['scale']);full=full_library(model,queries,v,index)
            row[split]={'cases':{'full':full,'local':local},'metrics':{'full':old_aggregate(full),'local':old_aggregate(local)}}
        rows.append(row)
    return rows


def execute():
    if (EXP/'protocol.json').exists():
        raise RuntimeError('本轮诊断已有执行记录，禁止自动覆盖')
    lock=json.loads((EXP/'design_lock.json').read_text(encoding='utf-8'))
    for name,h in lock['code_sha256'].items():
        if sha(ROOT/'scripts'/name)!=h:
            raise ValueError('诊断冻结代码发生变化')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    records,vectors,index=load()
    train=[q for q in records if q['split']=='train'];val=[q for q in records if q['split']=='validation']
    pca_indices=sorted({i for q in train for i in q['di']})
    center,basis=pca_fit(vectors['documents'],pca_indices)
    if len(basis)<max(DIMS):
        raise ValueError('训练候选文档不足以估计最大PCA维度')
    torch.save({'center':center,'basis':basis,'fit_indices':pca_indices},EXP/'pca_fit.pt')
    protocol={'scope':'仅训练与验证诊断；不读取测试查询向量或评分测试查询，不作新的盲测或曲率确认性结论。',
              'dimensions':DIMS,'seeds':SEEDS,'learning_rates':LRS,'steps':STEPS,'conditions':CONDITIONS,
              'selection':'全库验证分支宏R@1优先，MRR打破平手；包括第0步。局部训练另保存按局部验证选择的检查点，用于诊断选择偏差。',
              'pca':'在训练查询候选文档的标准化单位向量上拟合中心化PCA；不使用验证或测试查询标签拟合。',
              'random':'同一PCA文档中心，固定种子正交随机投影。PCA和随机投影都按余弦评分。',
              'training':'共享线性投影、cosine评分；比较10候选交叉熵和5000候选全softmax；所有文档参与全库训练梯度。',
              'precision':'新cosine诊断使用float32，禁用TF32；旧欧氏与双曲检查点仍按float64评分。',
              'device':str(vectors['documents'].device),'pca_fit_documents':len(pca_indices),
              'source_manifest_sha256':sha(DATA/'manifest.json'),'source_vector_sha256':{n:sha(EMBED/n) for n in ('documents.npy','queries.npy')},
              'code_sha256':lock['code_sha256'],'recovery_gate':'探索性条件：全库验证宏R@1至少达到原始cosine减2个百分点；不是统计显著性门槛。'}
    save(EXP/'protocol.json',protocol)
    rows=[]
    for name,raw in (('raw_cosine',True),('whitened_cosine',False)):
        rows.append(record(name,1536,None,None,train,val,vectors,raw=raw))
    rows+=old_checkpoint_diagnostic(train,val,vectors,index)
    for dim in DIMS:
        rows.append(record('pca_cosine',dim,None,CosineHead(basis[:dim],center),train,val,vectors))
        for seed in SEEDS:
            rb=random_basis(dim,seed,vectors['documents'].device,vectors['documents'].dtype)
            rows.append(record('random_cosine',dim,seed,CosineHead(rb,center),train,val,vectors))
    save(EXP/'baseline_results.json',{'summary':summarize(rows),'runs':rows})
    print('未训练的维度压缩基线与旧检查点诊断完成，仅涉及训练/验证查询。',flush=True)
    all_runs=[];registry=[]
    for dim in DIMS:
        for init,objective in CONDITIONS:
            name=init+'_'+objective
            for seed in SEEDS:
                start=basis[:dim] if init=='pca' else random_basis(dim,seed,vectors['documents'].device,vectors['documents'].dtype)
                best={}
                for lr in LRS:
                    chosen,run=fit(start,center,objective,seed,lr,train,val,vectors)
                    all_runs.append({'dimension':dim,'initialization':init,'objective':objective,**run,
                                     'selected':{c:{'step':chosen[c]['step'],'validation':{x:aggregate(chosen[c]['validation'][x]) for x in ('full','local')}} for c in chosen}})
                    for condition in ('full','local'):
                        if condition not in best or key(chosen[condition]['validation'],condition)>key(best[condition]['validation'],condition):
                            best[condition]={**chosen[condition],'lr':lr}
                for selector in ('full','local') if objective=='local' else ('full',):
                    selected=best[selector];model=CosineHead(start,center);model.load_state_dict(selected['state'])
                    label=name if selector=='full' else name+'_local_selection'
                    filename=f'{label}_{dim}_{seed}.pt'
                    torch.save(model.state_dict(),EXP/filename)
                    selection={'step':selected['step'],'lr':selected['lr'],'condition':selector,'checkpoint':filename,'sha256':sha(EXP/filename)}
                    registry.append({'name':label,'dimension':dim,'seed':seed,**selection})
                    rows.append(record(label,dim,seed,model,train,val,vectors,selection))
                save(EXP/'training_runs.json',all_runs);save(EXP/'selection_registry.json',registry)
                save(EXP/'results.json',{'stage':'诊断进行中','summary':summarize(rows),'runs':rows})
                print(f"维度{dim}，{name}，种子{seed}：全库验证宏R@1={key(best['full']['validation'])[0]:.1%}，选中步数{best['full']['step']}。",flush=True)
    summary=summarize(rows)
    reference=next(r for r in summary if r['name']=='raw_cosine')['validation']['full']['macro_r1']
    recovered=[{'name':r['name'],'dimension':r['dimension'],'validation_macro_r1':r['validation']['full']['macro_r1']}
               for r in summary if r['dimension']!=1536 and r['validation']['full']['macro_r1']>=reference-.02]
    checks={'only_train_validation':all(split in ('train','validation') for split in (q['split'] for q in records)),
            'all_checkpoints':all(sha(EXP/r['checkpoint'])==r['sha256'] for r in registry),
            'source_manifest':sha(DATA/'manifest.json')==protocol['source_manifest_sha256']}
    if not all(checks.values()):
        raise ValueError('诊断结束后的核验失败')
    save(EXP/'results.json',{'stage':'已完成','scope':protocol['scope'],'protocol_sha256':sha(EXP/'protocol.json'),
                            'summary':summary,'runs':rows,'recovery_gate':{'reference':reference,'tolerance':.02,'recovered':recovered},'checks':checks})
    lines=['# 冻结向量读出退化诊断','','仅训练/验证结果，已打开的测试查询没有再次评分；所有数值是探索性选择后的验证表现。','','| 读出 | 维度 | 训练全库宏R@1 | 验证全库宏R@1 | 验证10候选宏R@1 |','|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['name']} | {r['dimension']} | {r['train']['full']['macro_r1']:.1%} | {r['validation']['full']['macro_r1']:.1%} | {r['validation']['local']['macro_r1']:.1%} |")
    lines+=['','原始cosine全库验证宏R@1为'+f'{reference:.1%}'+f'；达到其减2个百分点探索性门槛的条件共{len(recovered)}项。',
            'PCA只拟合训练候选文档；随机投影与学习投影使用相同PCA中心。新cosine诊断为float32，旧距离检查点仍为float64。',
            '本轮只诊断压缩、负例及选择影响，不能证明泛化提升、曲率优势或替代新的盲测。','']
    (EXP/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('读出退化诊断完成；未评分任何测试查询。',flush=True)


def main():
    EXP.mkdir(parents=True,exist_ok=True)
    if (EXP/'status.json').exists() or (EXP/'protocol.json').exists():
        raise RuntimeError('已有诊断执行记录，禁止覆盖状态或自动重跑')
    save(EXP/'status.json',{'stage':'运行中','pid':os.getpid(),'utc':datetime.now(UTC).isoformat()})
    try:
        execute()
        save(EXP/'status.json',{'stage':'已完成','pid':os.getpid(),'utc':datetime.now(UTC).isoformat(),'results_sha256':sha(EXP/'results.json')})
    except Exception as error:
        save(EXP/'status.json',{'stage':'失败','pid':os.getpid(),'utc':datetime.now(UTC).isoformat(),'error':str(error)})
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
