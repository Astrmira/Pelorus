"""冻结程序生成的日期反事实文本；测试日期、物品与问法均隔离。"""

import hashlib
import json
from datetime import date,timedelta
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/date_counterfactual_v1'
EMBED=ROOT/'artifacts/embeddings/date_counterfactual_v1'
EXP=ROOT/'artifacts/experiments/date_counterfactual_v1'
INSTRUCTION='Instruct: Given a personal conversation history, retrieve the passages that provide the evidence needed to answer the question.\nQuery: '
WEEKDAYS=('Mon','Tue','Wed','Thu','Fri','Sat','Sun')


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build():
    if (DATA/'manifest.json').exists():raise RuntimeError('新日期反事实数据已冻结，禁止覆盖')
    rng=np.random.default_rng(20261016)
    nouns=('notebook','keychain','folder','badge','mug','backpack','scarf','journal','wallet','umbrella','pencil case','water bottle')
    subjects=[a+' '+n for a in ('travel','office','personal','spare','studio','weekend','school') for n in nouns]
    train_subjects=subjects[:48]
    val_subjects=subjects[48:60]
    test_subjects=subjects[60:84]
    used_dates=set()
    cases,documents,queries=[],[],[]
    for split,items in (('train',train_subjects),('validation',val_subjects),('interpolation',test_subjects),('extrapolation',test_subjects)):
        years=(2024,2025) if split=='extrapolation' else (2021,2022,2023)
        pool=[date(y,1,1)+timedelta(days=i) for y in years for i in range((date(y+1,1,1)-date(y,1,1)).days)]
        for number,subject in enumerate(items):
            same_year=number%2==0
            while True:
                chosen=rng.choice(len(pool),2,replace=False)
                early,late=sorted(pool[int(i)] for i in chosen)
                if (early.year==late.year)==same_year and (late-early).days>=30 and not {early,late}&used_dates:break
            used_dates.update((early,late))
            cid='dc_'+digest(split+'\0'+subject)[:20]
            case={'id':cid,'split':split,'subject':subject,'cluster':subject,'stratum':'same_year' if same_year else 'cross_year',
                  'dates':[early.isoformat(),late.isoformat()],'documents':[],'queries':[]}
            # 四个正文×日期组合使日期与颜色严格正交；两个候选条件均按颜色A/B固定位置。
            for when,colour in ((early,'navy'),(late,'scarlet'),(late,'navy'),(early,'scarlet')):
                text=f'Date: {when:%Y/%m/%d} ({WEEKDAYS[when.weekday()]}) 00:00\nUser: My {subject} is {colour}.'
                did='dc_d_'+digest(text)[:24]
                documents.append({'id':did,'text':text})
                case['documents'].append({'id':did,'date':when.isoformat(),'ordinal':when.toordinal(),'colour':colour})
            if split in ('train','validation'):
                templates=((f'Find the earlier dated update about my {subject}.',f'Find the later dated update about my {subject}.'),
                           (f'Retrieve the first update chronologically about my {subject}.',f'Retrieve the last update chronologically about my {subject}.'))
            else:
                templates=((f'Use the timestamps to select the initial message concerning my {subject}.',f'Use the timestamps to select the final message concerning my {subject}.'),
                           (f'Which report of my {subject} precedes the other report in time?',f'Which report of my {subject} follows the other report in time?'),
                           (f'Among these dated statements about my {subject}, return the oldest one.',f'Among these dated statements about my {subject}, return the newest one.'))
            def add(style,family,role,text):
                qid='dc_q_'+digest(cid+'\0'+style+str(family)+str(role))[:24]
                queries.append({'id':qid,'text':INSTRUCTION+text})
                case['queries'].append({'id':qid,'style':style,'family':family,'role':role,'text':text})
            for family,pair in enumerate(templates):
                for role,text in enumerate(pair):add('temporal',family,role,text)
            for role,colour in enumerate(('navy','scarlet')):
                add('ordinary',0,role,f'Find the report that says my {subject} is {colour}.')
            if split not in ('train','validation'):add('neutral',0,0,f'What did I report about my {subject}?')
            cases.append(case)
    for name,rows in (('documents',documents),('queries',queries)):
        (DATA).mkdir(parents=True,exist_ok=True)
        assert len({r['id'] for r in rows})==len(rows)
        (DATA/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    save(DATA/'annotations.json',cases)
    manifest={'chains':len(cases),'documents':len(documents),'queries':len(queries),
              'splits':{s:sum(c['split']==s for c in cases) for s in ('train','validation','interpolation','extrapolation')},
              'generator_sha256':sha(Path(__file__)),'label':'程序产生有效日期，主要标签仅按Date元数据先后；无人工事实标注。',
              'limitations':['这是合成文本诊断，不是自然用户数据或全库检索。','外推与插值测试复用物品主题，需要按物品配对，不能合并为48个独立主题。',
                             '插值测试也整体留出了日期值、物品描述及时间问法；测试不只改变一个因素。',
                             '二头探针使用额外的日期/查询类别监督，属于信息可读性诊断，不与检索监督模型作公平几何优越性主张。'],
              'input_sha256':{name:sha(DATA/name) for name in ('annotations.json','documents.jsonl','queries.jsonl')}}
    save(DATA/'manifest.json',manifest)
    config=yaml.safe_load((ROOT/'configs/temporal_fresh_test_encoding.yml').read_text(encoding='utf-8'))
    config['experiment']={'seed':20261016,'dataset_manifest':str((DATA/'manifest.json').relative_to(ROOT)),'output_dir':str(EMBED.relative_to(ROOT))}
    config['inputs']={k:str((DATA/(k+'.jsonl')).relative_to(ROOT)) for k in ('documents','queries')}
    config['embedding']['batch_size']=32
    for key,value in config['cache'].items():config['cache'][key]=str((EMBED/Path(value).name).relative_to(ROOT))
    (ROOT/'configs/date_counterfactual_encoding.yml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('chains','documents','queries','splits')},ensure_ascii=False))


if __name__=='__main__':build()
