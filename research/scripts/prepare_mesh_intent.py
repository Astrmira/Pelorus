"""用真实MeSH定义与全部直接父关系构造概念／父类意图检索。"""

import argparse
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml

from prepare_pubmed_geometry import ROOT,sha,save,write_jsonl

SOURCE=ROOT/'data/raw/pubmed_geometry_v1/desc2026.gz'
DATA=ROOT/'data/processed/mesh_intent_geometry_v1'
EMBED=ROOT/'artifacts/embeddings/mesh_intent_geometry_v1'
EXP=ROOT/'artifacts/experiments/mesh_intent_geometry_v1'
INSTRUCTION='Instruct: Retrieve the MeSH concept entry requested by the query, respecting whether it asks for the concept itself or a direct broader category.\nQuery: '
TEMPLATES={
    'train':(
        ('Find the MeSH entry that defines {term} itself.','Find a direct broader MeSH category of {term}.'),
        ('Retrieve the definition of the concept {term}.','Retrieve an immediate parent category of {term}.'),
        ('Which vocabulary entry describes {term}?','Which descriptor is one level above {term} in the MeSH hierarchy?'),
        ('Return the descriptor for {term} itself.','Return a direct superclass of {term}.')),
    'validation':(
        ('Select the entry for {term} itself.','Select an immediate superordinate term for {term}.'),
        ('I need the definition of {term}.','I need a direct parent category of {term}.')),
    'test':(
        ('Locate the descriptor corresponding to {term}.','Locate a descriptor immediately broader than {term}.'),
        ('Which entry explains {term} as a concept?','Which entry names a direct parent of {term}?'))}


def parse_mesh(body):
    entries={}
    for r in ET.fromstring(gzip.decompress(body)).findall('DescriptorRecord'):
        uid=r.findtext('DescriptorUI')
        notes=[c.findtext('ScopeNote','') for c in r.findall('ConceptList/Concept') if c.attrib.get('PreferredConceptYN')=='Y']
        entries[uid]={'id':uid,'name':r.findtext('DescriptorName/String'),
                      'definition':' '.join((notes[0] if notes else '').split()),
                      'trees':[t.text for t in r.findall('TreeNumberList/TreeNumber')]}
    tree_ids={t:r['id'] for r in entries.values() for t in r['trees']}
    for uid,r in entries.items():
        r['parents']=sorted({tree_ids[t.rsplit('.',1)[0]] for t in r['trees'] if '.' in t and t.rsplit('.',1)[0] in tree_ids}-{uid})
        r['branches']=sorted({'.'.join(t.split('.')[:3]) for t in r['trees'] if t.startswith('C') and len(t.split('.'))>=3})
    return entries


def sample_data(entries,counts=None,pool_size=5000,seed=20261016):
    counts=counts or {'train':300,'validation':100,'test':200}
    rng=np.random.default_rng(seed)
    documents={k:v for k,v in entries.items() if v['name'] and len(v['definition'])>=40}
    eligible={uid:r for uid,r in documents.items() if r['branches'] and any(t.startswith('C') and len(t.split('.'))>=4 for t in r['trees'])
              and r['parents'] and all(p in documents for p in r['parents'])}
    branches=sorted({b for r in eligible.values() for b in r['branches']})
    ordered=[branches[int(i)] for i in rng.permutation(len(branches))]
    n=len(ordered);a=int(n*.5);b=int(n*.2)
    assignments={branch:split for split,items in (('train',ordered[:a]),('validation',ordered[a:a+b]),('test',ordered[a+b:])) for branch in items}
    available={s:[] for s in counts}
    for r in eligible.values():
        splits={assignments[b] for b in r['branches']}
        if len(splits)==1:
            available[next(iter(splits))].append(r)
    # 任何留出焦点概念都不能已作为较早划分的父类正例出现。
    train_targets={p for r in available['train'] for p in r['parents']}
    available['validation']=[r for r in available['validation'] if r['id'] not in train_targets]
    val_targets={p for r in available['validation'] for p in r['parents']}
    available['test']=[r for r in available['test'] if r['id'] not in train_targets|val_targets]
    coverage={'eligible':len(eligible),'branch_count':len(branches),
              'available':{s:{'total':len(rows),'multi_parent':sum(len(r['parents'])>1 for r in rows)} for s,rows in available.items()}}
    focus=[]
    for split,required in counts.items():
        chosen=[]
        for multi,quota in ((True,required//4),(False,required-required//4)):
            groups={}
            for r in available[split]:
                if (len(r['parents'])>1)==multi:
                    groups.setdefault(r['branches'][0],[]).append(r)
            if sum(map(len,groups.values()))<quota:
                raise ValueError('分支隔离后概念或多父标签覆盖不足：'+json.dumps(coverage,ensure_ascii=False))
            for rows in groups.values():
                rng.shuffle(rows)
            subset=[]
            while len(subset)<quota:
                for branch in sorted(groups):
                    if groups[branch] and len(subset)<quota:
                        subset.append(groups[branch].pop())
            chosen+=subset
        focus += [{**r,'split':split,'cluster':r['branches'][0]} for r in chosen]
    reserved={r['id'] for r in focus}|{p for r in focus for p in r['parents']}
    if len(reserved)>pool_size or len(documents)<pool_size:
        raise ValueError('固定文档库容量不满足概念与父类覆盖要求')
    others=[uid for uid in sorted(documents) if uid not in reserved]
    ids=sorted(reserved)+[others[int(i)] for i in rng.permutation(len(others))[:pool_size-len(reserved)]]
    rng.shuffle(ids)
    docs=[documents[uid] for uid in ids]
    queries=[]
    for r in focus:
        for family,pair in enumerate(TEMPLATES[r['split']]):
            for intent,template in zip(('self','parent'),pair):
                text=template.format(term=r['name'])
                queries.append({'id':f"mi_{r['split']}_{r['id']}_{family}_{intent}",'text':INSTRUCTION+text,
                                'focus_id':r['id'],'split':r['split'],'cluster':r['cluster'],'family':family,'intent':intent,
                                'positive_ids':[r['id']] if intent=='self' else r['parents']})
    coverage.update(focus_counts={s:sum(r['split']==s for r in focus) for s in counts},
                    focus_clusters={s:len({r['cluster'] for r in focus if r['split']==s}) for s in counts},
                    query_counts={s:sum(q['split']==s for q in queries) for s in counts},documents=len(docs))
    return docs,focus,queries,coverage


def validate(entries,docs,focus,queries):
    ids={d['id'] for d in docs}
    if len(ids)!=len(docs) or len({q['id'] for q in queries})!=len(queries):
        raise ValueError('文档或查询ID重复')
    branch_sets={s:{b for r in focus if r['split']==s for b in r['branches']} for s in ('train','validation','test')}
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        if branch_sets[a]&branch_sets[b]:
            raise ValueError('疾病树分支跨划分重叠')
        earlier={p for r in focus if r['split']==a for p in [r['id'],*r['parents']]}
        later={r['id'] for r in focus if r['split']==b}
        if earlier&later:
            raise ValueError('留出焦点已作为较早划分的正例出现')
    for q in queries:
        expected=[q['focus_id']] if q['intent']=='self' else entries[q['focus_id']]['parents']
        if set(q['positive_ids'])!=set(expected) or not set(expected)<=ids:
            raise ValueError('未完整保留直接父类或目标缺失')
        if q['focus_id'] not in ids:
            raise ValueError('概念自身未保留在候选库')
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    if (DATA/'manifest.json').exists():
        raise RuntimeError('层级意图数据已冻结，禁止覆盖')
    entries=parse_mesh(SOURCE.read_bytes())
    docs,focus,queries,coverage=sample_data(entries)
    validate(entries,docs,focus,queries)
    print(json.dumps(coverage,ensure_ascii=False,indent=2),flush=True)
    if args.audit_only:
        return
    DATA.mkdir(parents=True,exist_ok=True)
    write_jsonl(DATA/'documents.jsonl',[{'id':r['id'],'text':f"MeSH term: {r['name']}\nScope note: {r['definition']}"} for r in docs])
    write_jsonl(DATA/'queries.jsonl',[{'id':q['id'],'text':q['text']} for q in queries])
    save(DATA/'annotations.json',queries);save(DATA/'focus_metadata.json',focus);save(DATA/'document_metadata.json',docs)
    examples=[]
    for split in ('train','validation'):
        selected=[r for r in focus if r['split']==split]
        for multi in (False,True):
            for r in [r for r in selected if (len(r['parents'])>1)==multi][:2]:
                examples.append({'focus':r,'parents':[entries[p] for p in r['parents']],
                                 'queries':[q for q in queries if q['focus_id']==r['id']]})
    save(DATA/'review_examples.json',examples)
    manifest={'source':'NLM MeSH 2026官方描述词XML','source_sha256':sha(SOURCE),'coverage':coverage,
              'generator_sha256':sha(Path(__file__)),'labels':'概念自身或全部直接父类；所有树路径的有效直接父类均为正例。',
              'separation':'焦点概念的全部疾病树前三段分支仅归属同一划分；留出焦点不作为较早划分正例。',
              'limitations':['真实专家定义与关系上的模板查询，属于机制诊断，不是自然用户问题。',
                             '保留疾病树分支隔离，但其他树路径、共享上位文档与语义可能跨划分。',
                             '只检索任意一个有效直接父类别算Top-1正确；另报父集合Recall@5。',
                             '训练关系标签提供额外监督，收益不能直接解释为无监督换几何。',
                             '文档库包含全部焦点与父类别后补齐5000篇，不是总体随机语料库。'],
              'input_sha256':{n:sha(DATA/n) for n in ('documents.jsonl','queries.jsonl','annotations.json','focus_metadata.json','document_metadata.json')}}
    save(DATA/'manifest.json',manifest)
    config=yaml.safe_load((ROOT/'configs/pubmed_geometry_encoding.yml').read_text(encoding='utf-8'))
    config['experiment']={'seed':20261016,'dataset_manifest':'data/processed/mesh_intent_geometry_v1/manifest.json','output_dir':'artifacts/embeddings/mesh_intent_geometry_v1'}
    config['inputs']={k:f'data/processed/mesh_intent_geometry_v1/{k}.jsonl' for k in ('documents','queries')}
    for k,v in config['cache'].items():
        config['cache'][k]='artifacts/embeddings/mesh_intent_geometry_v1/'+v.replace('\\','/').split('/')[-1]
    (ROOT/'configs/mesh_intent_encoding.yml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    save(EXP/'encoding_lock.json',{'stage':'向量编码、训练与评分之前冻结','manifest_sha256':sha(DATA/'manifest.json'),
                                 'config_sha256':sha(ROOT/'configs/mesh_intent_encoding.yml'),
                                 'code_sha256':{n:sha(ROOT/'scripts'/n) for n in ('prepare_mesh_intent.py','run_mesh_intent.py','mesh_intent_pipeline.py','geometry_readouts.py')}})


if __name__=='__main__':
    main()
