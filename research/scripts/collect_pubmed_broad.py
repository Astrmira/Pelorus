"""纠正PubMed疾病检索词自动映射过窄的问题；保留首轮原始记录。"""

import json
import time
import urllib.parse
import xml.etree.ElementTree as ET

from collect_pubmed_geometry import RAW,fetch,sha


def main():
    base=RAW/'broad_medline'
    base.mkdir(parents=True,exist_ok=True)
    descriptors=json.loads((RAW/'mesh_descriptors.json').read_text(encoding='utf-8'))
    term='2019:2023[dp] AND hasabstract AND medline[sb]'
    search='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'+urllib.parse.urlencode(
        {'db':'pubmed','term':term,'retmax':9999,'retmode':'json','sort':'pub date','tool':'quiver_research'})
    ids=json.loads(fetch(search,base/'search.json'))['esearchresult']['idlist']
    records=[]
    for start in range(0,len(ids),200):
        url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'+urllib.parse.urlencode(
            {'db':'pubmed','id':','.join(ids[start:start+200]),'retmode':'xml','tool':'quiver_research'})
        xml=ET.fromstring(fetch(url,base/f'pubmed_{start:05d}.xml'))
        for article in xml.findall('PubmedArticle'):
            citation=article.find('MedlineCitation')
            title=citation.find('Article/ArticleTitle')
            abstract=citation.findall('Article/Abstract/AbstractText')
            title='' if title is None else ''.join(title.itertext()).strip()
            text='\n'.join((a.attrib.get('Label','')+': ' if a.attrib.get('Label') else '')+''.join(a.itertext()).strip() for a in abstract)
            mesh=[{'id':h.attrib['UI'],'major':h.attrib.get('MajorTopicYN')=='Y'} for h in citation.findall('MeshHeadingList/MeshHeading/DescriptorName') if h.attrib.get('UI') in descriptors]
            if title and 400<=len(text)<=6500 and mesh:
                records.append({'pmid':citation.findtext('PMID'),'title':title,'abstract':text,'mesh':mesh})
        print(f'广域MEDLINE已解析{min(start+200,len(ids))}/{len(ids)}篇，合格记录{len(records)}篇。',flush=True)
        time.sleep(.4)
    unique={r['pmid']:r for r in records}
    (base/'articles.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in unique.values()),encoding='utf-8')
    manifest={'pubmed_search':search,'search_term':term,'records':len(unique),
              'selection':'2019–2023已索引MEDLINE有摘要文章，按发布日期排序取前9999篇；再与首轮疾病主题记录合并去重；不是随机总体样本。',
              'reason':'首轮diseases[mh]被PubMed自动映射为单个Disease主题；仅在元数据阶段扩大覆盖，尚未编码或查看模型分数。',
              'files':{p.name:sha(p) for p in sorted(base.iterdir()) if p.is_file() and p.name!='source_manifest.json'}}
    (base/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('广域MEDLINE收集完成。',flush=True)


if __name__=='__main__':
    main()
