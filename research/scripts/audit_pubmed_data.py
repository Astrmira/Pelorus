"""核验本轮真实数据的全部候选关系；不读取向量或模型分数。"""

import json
from prepare_pubmed_geometry import ROOT,RAW,DATA,EXP,save,sha


def main():
    queries=json.loads((DATA/'annotations.json').read_text(encoding='utf-8'))
    docs=json.loads((DATA/'document_metadata.json').read_text(encoding='utf-8'))
    index={d['id']:d for d in docs}
    mesh=json.loads((RAW/'mesh_descriptors.json').read_text(encoding='utf-8'))
    clusters={s:{r['cluster'] for r in queries if r['split']==s} for s in ('train','validation','test')}
    assert all(not clusters[a]&clusters[b] for a,b in (('train','validation'),('train','test'),('validation','test')))
    assert len(index)==5000 and len({r['id'] for r in queries})==500
    for r in queries:
        assert len(r['candidates'])==len({c['id'] for c in r['candidates']})==10
        assert r['candidates'][r['target_index']]['id']==r['target_id']
        assert r['anchor_tree'] in mesh[r['anchor_mesh']]['trees']
        parent=r['anchor_tree'].rsplit('.',1)[0]
        for candidate in r['candidates']:
            d=index[candidate['id']]
            if candidate['type']=='parent':
                assert d['anchor_tree']==parent
            if candidate['type']=='grandparent':
                assert d['anchor_tree']==parent.rsplit('.',1)[0]
            if candidate['type']=='sibling':
                assert d['anchor_tree']!=r['anchor_tree'] and d['anchor_tree'].rsplit('.',1)[0]==parent
            if candidate['type'] in ('parent','grandparent','sibling'):
                assert r['anchor_mesh'] not in {h['id'] for h in d['mesh']}
    review={'passed':True,'manifest_sha256':sha(DATA/'manifest.json'),
            'checks':['500个查询与5000文档ID唯一','目标位置一致','规范分支划分互斥','候选层级关系与真实MeSH树一致','困难负例不携带目标锚点标签'],
            'reviewer':'助手文本抽查，非医学专家标注',
            'text_spot_checks':'会话中已检查2条训练及2条验证查询，合计20个目标和层级困难候选；未查看模型分数。',
            'important_limit':'层级类别是MeSH规范锚点的关系，不保证整篇摘要的语义广度；评测目标是标题对应文章。',
            'topics':{s:len(ts) for s,ts in clusters.items()},'audit_script_sha256':sha(ROOT/'scripts/audit_pubmed_data.py')}
    save(EXP/'data_review.json',review)
    print(json.dumps(review,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
