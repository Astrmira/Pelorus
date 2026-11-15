"""按真实疾病本体分支构造标题到摘要检索；在编码与训练前冻结。"""

import hashlib
import json
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/pubmed_geometry_v1'
DATA=ROOT/'data/processed/pubmed_geometry_v1'
EMBED=ROOT/'artifacts/embeddings/pubmed_geometry_v1'
EXP=ROOT/'artifacts/experiments/pubmed_geometry_v1'
INSTRUCTION='Instruct: Given a biomedical article title, retrieve its corresponding abstract.\nQuery: '


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_jsonl(path,rows):
    # 真实文章包含Unicode分隔符；转义后兼容旧编码器的splitlines读取，文本内容保持一致。
    path.write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in rows),encoding='utf-8')


def build(records,descriptors,pool_size=5000,counts=None,seed=20261016):
    counts=counts or {'train':200,'validation':100,'test':200}
    rng=np.random.default_rng(seed)
    unique={}
    for r in records:
        # 去除重复标题与摘要；选择最深的主要疾病MeSH，若没有主要疾病标签则使用所有疾病标签。
        choices=[(len(tree.split('.')),tree,h['id']) for h in r['mesh'] for tree in descriptors[h['id']]['trees'] if tree.startswith('C') and h['major']]
        if not choices:
            choices=[(len(tree.split('.')),tree,h['id']) for h in r['mesh'] for tree in descriptors[h['id']]['trees'] if tree.startswith('C')]
        if not choices:
            continue
        _,tree,uid=sorted(choices,key=lambda x:(-x[0],x[1],x[2]))[0]
        article={**r,'id':'pm_'+r['pmid'],'anchor_tree':tree,'anchor_mesh':uid,'topic':'.'.join(tree.split('.')[:3])}
        text_key=(r['title'].strip().lower(),r['abstract'].strip().lower())
        if not any(k in unique for k in text_key):
            for k in text_key:
                unique[k]=article
    articles={r['id']:r for r in unique.values()}
    ordered=[articles[k] for k in sorted(articles)]
    if len(ordered)<pool_size:
        raise ValueError(f'符合疾病锚点条件的唯一文章不足：{len(ordered)}，需要{pool_size}')
    by_tree={}
    for r in ordered:
        by_tree.setdefault(r['anchor_tree'],[]).append(r)
    viable=[]
    for r in ordered:
        path=r['anchor_tree'].split('.')
        if len(path)<4:
            continue
        parent='.'.join(path[:-1]);grandparent='.'.join(path[:-2])
        siblings=[d for t,docs in by_tree.items() if t!=r['anchor_tree'] and '.'.join(t.split('.')[:-1])==parent for d in docs]
        # 干扰文档不得同时携带目标锚点标签，以免把多标签文章强行视为上位类别负例。
        clean=lambda rows:[d for d in rows if r['anchor_mesh'] not in {h['id'] for h in d['mesh']}]
        categories={'sibling':clean(siblings),'parent':clean(by_tree.get(parent,[])),
                    'grandparent':clean(by_tree.get(grandparent,[]))}
        if len(categories['sibling'])>=2 and categories['parent'] and categories['grandparent']:
            viable.append((r,categories))
    # 分支分割只依赖元数据；失败时报告覆盖不足，不按模型分数改变样本。
    topics=sorted({r['topic'] for r,_ in viable})
    perm=[topics[int(i)] for i in rng.permutation(len(topics))]
    n=len(perm);ntrain=int(n*.45);nval=int(n*.2)
    split_topics={'train':set(perm[:ntrain]),'validation':set(perm[ntrain:ntrain+nval]),'test':set(perm[ntrain+nval:])}
    candidates={s:[(r,c) for r,c in viable if r['topic'] in ts] for s,ts in split_topics.items()}
    coverage={'source_eligible':len(ordered),'pool':pool_size,'viable':len(viable),'topics':len(topics),
              'split_available':{s:len(rows) for s,rows in candidates.items()},
              'split_topics':{s:len(ts) for s,ts in split_topics.items()}}
    for s,required in counts.items():
        if len(candidates[s])<required:
            raise ValueError('分支留出后候选不足：'+json.dumps(coverage,ensure_ascii=False))
    queries=[]
    for split,required in counts.items():
        groups={t:[] for t in split_topics[split]}
        for r,c in candidates[split]:
            groups[r['topic']].append((r,c))
        for rows in groups.values():
            rng.shuffle(rows)
        chosen=[]
        # 轮流抽取各分支，避免少数常见疾病占满查询。
        while len(chosen)<required:
            for t in sorted(groups):
                if groups[t] and len(chosen)<required:
                    chosen.append(groups[t].pop())
        for r,categories in chosen:
            items=[{'id':r['id'],'type':'target'}]
            for label,count in (('sibling',2),('parent',1),('grandparent',1)):
                rows=categories[label]
                items += [{'id':rows[int(i)]['id'],'type':label} for i in rng.choice(len(rows),count,replace=False)]
            excluded={i['id'] for i in items}
            unrelated=[d for d in ordered if d['id'] not in excluded and d['topic']!=r['topic'] and r['anchor_mesh'] not in {h['id'] for h in d['mesh']}]
            items += [{'id':unrelated[int(i)]['id'],'type':'random'} for i in rng.choice(len(unrelated),5,replace=False)]
            rng.shuffle(items)
            queries.append({'id':'pq_'+r['pmid'],'text':INSTRUCTION+r['title'],'split':split,'cluster':r['topic'],
                            'target_id':r['id'],'anchor_tree':r['anchor_tree'],'anchor_mesh':r['anchor_mesh'],
                            'candidates':items,'target_index':next(i for i,d in enumerate(items) if d['id']==r['id'])})
    # 先保留元数据构造的目标与困难候选，再随机补齐文档池；避免随机抽库把稀有祖先文档丢掉。
    reserved={d['id'] for q in queries for d in q['candidates']}
    if len(reserved)>pool_size:
        raise ValueError('已选候选超过固定文档池容量')
    others=[d for d in ordered if d['id'] not in reserved]
    pool=[articles[i] for i in sorted(reserved)]+[others[int(i)] for i in rng.permutation(len(others))[:pool_size-len(reserved)]]
    rng.shuffle(pool)
    return pool,queries,coverage


def main():
    if (DATA/'manifest.json').exists():
        raise RuntimeError('真实语料数据集已经冻结，禁止覆盖')
    descriptors=json.loads((RAW/'mesh_descriptors.json').read_text(encoding='utf-8'))
    records=[json.loads(line) for line in (RAW/'articles.jsonl').read_text(encoding='utf-8').split('\n') if line]
    records += [json.loads(line) for line in (RAW/'broad_medline/articles.jsonl').read_text(encoding='utf-8').split('\n') if line]
    pool,queries,coverage=build(records,descriptors)
    DATA.mkdir(parents=True,exist_ok=True)
    for kind,rows in (('documents',[{'id':r['id'],'text':r['abstract']} for r in pool]),
                      ('queries',[{'id':q['id'],'text':q['text']} for q in queries])):
        write_jsonl(DATA/(kind+'.jsonl'),rows)
    save(DATA/'annotations.json',queries)
    save(DATA/'document_metadata.json',[{k:v for k,v in r.items() if k!='abstract'} for r in pool])
    # 人工审计样本只取训练和验证；测试原始数据可以验证结构，但不用于选模型。
    examples=[]
    index={r['id']:r for r in pool}
    for split in ('train','validation'):
        for q in [q for q in queries if q['split']==split][:8]:
            examples.append({**q,'candidate_records':[{**d,'article':index[d['id']]} for d in q['candidates']]})
    save(DATA/'review_examples.json',examples)
    manifest={'seed':20261016,'coverage':coverage,'queries':len(queries),'documents':len(pool),
              'splits':{s:sum(q['split']==s for q in queries) for s in ('train','validation','test')},
              'source_manifest_sha256':{'initial':sha(RAW/'source_manifest.json'),'broad':sha(RAW/'broad_medline/source_manifest.json')},'generator_sha256':sha(Path(__file__)),
              'primary':'10候选已知文章检索R@1，按规范MeSH锚点的前三段分支聚类；不是一般生物医学相关性评测。',
              'secondary':'全部5000摘要中的已知文章检索R@1、MRR；困难候选错误类型。',
              'limitations':['只有标题对应摘要被定义为相关；其他文章即使主题相关也不是目标。',
                             '文档池先保留层级困难候选再补齐5000篇，故不是随机语料库。',
                             '候选类别按规范锚点定义，不保证摘要语义只属于一个层级。',
                             '规范锚点分支留出不保证所有次要MeSH标签或语义主题完全隔离。',
                             '候选文档全库可见，但只用训练查询相关性标签；不以本体边训练模型。'],
              'input_sha256':{n:sha(DATA/n) for n in ('documents.jsonl','queries.jsonl','annotations.json','document_metadata.json')}}
    save(DATA/'manifest.json',manifest)
    config=yaml.safe_load((ROOT/'configs/date_counterfactual_encoding.yml').read_text(encoding='utf-8'))
    config['experiment']={'seed':20261016,'dataset_manifest':'data/processed/pubmed_geometry_v1/manifest.json',
                          'output_dir':'artifacts/embeddings/pubmed_geometry_v1'}
    config['inputs']={k:f'data/processed/pubmed_geometry_v1/{k}.jsonl' for k in ('documents','queries')}
    for k,v in config['cache'].items():
        config['cache'][k]='artifacts/embeddings/pubmed_geometry_v1/'+v.replace('\\','/').split('/')[-1]
    config['embedding']['requests_per_minute']=20
    config['retry']['max_attempts']=5
    (ROOT/'configs/pubmed_geometry_encoding.yml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    save(EXP/'encoding_lock.json',{'stage':'任何新向量或模型评分之前冻结',
                                 'manifest_sha256':sha(DATA/'manifest.json'),
                                 'code_sha256':{n:sha(ROOT/'scripts'/n) for n in ('prepare_pubmed_geometry.py','run_pubmed_geometry.py','geometry_readouts.py','pubmed_cloud_pipeline.py')},
                                 'config_sha256':sha(ROOT/'configs/pubmed_geometry_encoding.yml')})
    print(json.dumps({'数据覆盖':coverage,'划分':manifest['splits']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
