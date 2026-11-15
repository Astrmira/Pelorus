"""下载真实MeSH与PubMed记录；仅按元数据收集，不读取向量或测试分数。"""

import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/pubmed_geometry_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url,path,form=None):
    if path.exists():
        return path.read_bytes()
    data=None if form is None else urllib.parse.urlencode(form).encode()
    for attempt in range(6):
        try:
            request=urllib.request.Request(url,data=data,headers={'User-Agent':'QuIVer-Research/1.0'})
            with urllib.request.urlopen(request,timeout=90) as response:
                body=response.read()
            if not body:
                raise ValueError('数据源返回空内容')
            path.write_bytes(body)
            return body
        except Exception:
            if attempt==5:
                raise
            time.sleep(min(2**attempt,30))


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    mesh_url='https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz'
    mesh_body=fetch(mesh_url,RAW/'desc2026.gz')
    descriptors={}
    for record in ET.fromstring(gzip.decompress(mesh_body)).findall('DescriptorRecord'):
        uid=record.findtext('DescriptorUI')
        descriptors[uid]={'name':record.findtext('DescriptorName/String'),
                          'trees':[t.text for t in record.findall('TreeNumberList/TreeNumber')]}
    (RAW/'mesh_descriptors.json').write_text(json.dumps(descriptors,ensure_ascii=False),encoding='utf-8')
    term='2019:2023[dp] AND hasabstract AND medline[sb] AND diseases[mh]'
    search_url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'+urllib.parse.urlencode(
        {'db':'pubmed','term':term,'retmax':9999,'retmode':'json','sort':'pub date','tool':'quiver_research'})
    ids=json.loads(fetch(search_url,RAW/'search.json'))['esearchresult']['idlist']
    records=[]
    for start in range(0,len(ids),200):
        path=RAW/f'pubmed_{start:05d}.xml'
        # 批量GET避免部分代理对POST请求的限制；每次最多200个PMID。
        url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'+urllib.parse.urlencode(
            {'db':'pubmed','id':','.join(ids[start:start+200]),'retmode':'xml','tool':'quiver_research'})
        document=ET.fromstring(fetch(url,path))
        for article in document.findall('PubmedArticle'):
            citation=article.find('MedlineCitation')
            title=citation.find('Article/ArticleTitle')
            abstract=citation.findall('Article/Abstract/AbstractText')
            title='' if title is None else ''.join(title.itertext()).strip()
            text='\n'.join((a.attrib.get('Label','')+': ' if a.attrib.get('Label') else '')+''.join(a.itertext()).strip() for a in abstract)
            mesh=[]
            for heading in citation.findall('MeshHeadingList/MeshHeading/DescriptorName'):
                if heading.attrib.get('UI') in descriptors:
                    mesh.append({'id':heading.attrib['UI'],'major':heading.attrib.get('MajorTopicYN')=='Y'})
            if title and 400<=len(text)<=6500 and mesh:
                records.append({'pmid':citation.findtext('PMID'),'title':title,'abstract':text,'mesh':mesh})
        print(f'已解析{min(start+200,len(ids))}/{len(ids)}篇原始文章，合格记录{len(records)}篇。',flush=True)
        time.sleep(.4)
    unique={r['pmid']:r for r in records}
    (RAW/'articles.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in unique.values()),encoding='utf-8')
    manifest={'sources':{'mesh':mesh_url,'pubmed_search':search_url},'search_term':term,
              'raw_articles':len(ids),'eligible_articles':len(unique),
              'selection':'发布日期2019–2023、MEDLINE已索引、疾病主题、有摘要；按PubMed日期排序取前9999篇；不是总体随机样本。',
              'files':{p.name:sha(p) for p in sorted(RAW.iterdir()) if p.is_file() and p.name!='source_manifest.json'}}
    (RAW/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('真实语料收集完成。',flush=True)


if __name__=='__main__':
    main()
