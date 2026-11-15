"""日期探针的训练标签打乱负对照；只作描述性机制诊断。"""

import json
from pathlib import Path
import numpy as np
import torch

from run_date_counterfactual import load,predict_ridge
from prepare_date_counterfactual import ROOT,EXP,save,sha


def main():
    lock=json.loads((EXP/'control_lock.json').read_text(encoding='utf-8'))
    if lock['sha256']!=sha(Path(__file__)):raise ValueError('负对照代码在冻结后发生变化')
    if (EXP/'label_shuffle_control.json').exists():raise RuntimeError('负对照已完成，禁止覆盖')
    torch.set_num_threads(1)
    cases,corpus,_,_=load()
    train=[c for c in cases if c['split']=='train']
    x=corpus['documents'][[i for c in train for i in c['di']]]
    y=np.array([[d['ordinal'] for d in c['documents']] for c in train])
    selected=json.loads((EXP/'head_selection.json').read_text(encoding='utf-8'))
    alpha=selected['date_alpha']
    fitted=torch.load(EXP/'scalar_heads.pt',map_location=x.device,weights_only=True)['date']
    rng=np.random.default_rng(20261017)
    labels=[]
    for _ in range(199):
        new=y[rng.permutation(len(train))].copy()
        reverse=rng.integers(0,2,len(train)).astype(bool)
        # 每个日期仍对应两种颜色；随机交换日期方向，避免保留早晚标签相关性。
        new[reverse]=new[reverse][:,[1,0,3,2]]
        labels.append(new.reshape(-1)/365.)
    ys=torch.tensor(np.stack(labels,axis=1),dtype=x.dtype,device=x.device)
    xc=x-x.mean(0);mean_y=ys.mean(0)
    weights=xc.T@torch.linalg.solve(xc@xc.T+alpha*torch.eye(len(x),dtype=x.dtype,device=x.device),ys-mean_y)
    result={}
    with torch.no_grad():
        for split in ('interpolation','extrapolation'):
            subset=[c for c in cases if c['split']==split]
            test=corpus['documents'][[i for c in subset for i in c['di']]]
            true=predict_ridge(fitted,test).reshape(len(subset),4)
            success=(true[:,0]<true[:,1])&(true[:,2]>true[:,3])
            predictions=((test-x.mean(0))@weights+mean_y).reshape(len(subset),4,199)
            null_success=(predictions[:,0]<predictions[:,1])&(predictions[:,2]>predictions[:,3])
            values=null_success.double().mean(0).cpu().numpy()
            actual=float(success.double().mean())
            result[split]={'true_labels_strict_date_order':actual,'null_mean':float(values.mean()),
                           'null_95pct_range':np.quantile(values,[.025,.975]).tolist(),
                           'null_at_least_as_good_fraction':float(np.mean(values>=actual)),
                           'null_runs':values.tolist()}
    save(EXP/'label_shuffle_control.json',{'code_sha256':lock['sha256'],'replicates':199,'alpha':alpha,
                                         'interpretation':'固定真实验证选定alpha的训练标签打乱；未给每个随机标签模型重新选alpha，因此尾部比例仅描述性，不当作确认性p值。',
                                         'results':result})
    print(json.dumps({k:{a:b for a,b in r.items() if a!='null_runs'} for k,r in result.items()},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
