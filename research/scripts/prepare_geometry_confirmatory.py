"""按原文冻结50条新证据链及确认性评估协议；不读取任何新测试指标。"""

import json
from pathlib import Path
import numpy as np
import yaml

from prepare_low_capacity_training_data import INSTRUCTION
from run_geometry_controlled import save, sha
from prepare_temporal_fresh_test import sha as text_sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/geometry_confirmatory_v1'
EXP = ROOT / 'artifacts/experiments/geometry_confirmatory_v1'
EMBED = ROOT / 'artifacts/embeddings/geometry_confirmatory_v1'
DEV = ROOT / 'artifacts/experiments/temporal_geometry_development_v1'

# 字段：来源题、证据索引、保守主题组、无答案主题、两条独有普通事实片段。
# 索引沿用阅读包；最终时间标签由严格递增的记录日期确定，不当作事件发生日期。
REVIEWED = [
 ('gpt4_59149c77',0,1,'museums','my museum visits','Museum of Modern Art','Metropolitan Museum of Art'),
 ('gpt4_f49edff3',0,2,'gifts','the gifts and preparations I described','nursery','customized phone case'),
 ('b46e15ed',1,2,'charity','my charity activities','24-Hour Bike Ride','Books for Kids'),
 ('gpt4_fa19884c',0,1,'music','my musical activities','old keyboard','bluegrass band'),
 ('0bc8ad92',0,2,'museums','my museum outings','chemistry professor','T-Rex'),
 ('9a707b81',0,1,'baking','my baking activities','strawberry shortcake','chocolate cake'),
 ('gpt4_1d4ab0c9',0,1,'gardening','my herb gardening progress','started watering','harvested my first batch'),
 ('gpt4_1d80365e',0,1,'travel','my solo Yosemite camping trip','started my solo camping trip','got back from an amazing solo camping trip'),
 ('gpt4_7f6b06db',0,2,'travel','my outdoor trips','Muir Woods','Yosemite National Park'),
 ('gpt4_18c2b244',0,2,'shopping','my shopping savings activities','Walmart','ShopRite'),
 ('gpt4_a1b77f9c',0,1,'reading','my reading of The Nightingale','started reading','finished reading'),
 ('gpt4_1916e0ea',0,1,'shopping','my grocery shopping activities','FarmFresh','Instacart'),
 ('gpt4_1e4a8aeb',0,1,'gardening','my gardening activities','gardening workshop','tomato saplings'),
 ('4dfccbf7',0,1,'music','my activities with musical instruments','ukulele lessons','guitar tech'),
 ('gpt4_61e13b3c',0,1,'markets','my participation in markets','homemade baked goods','handmade jewelry'),
 ('370a8ff4',0,1,'exercise','my recovery and exercise progress','recovered from the flu','10th jog'),
 ('gpt4_d6585ce8',0,4,'music','the concerts I attended','Billie Eilish','Queen'),
 ('gpt4_4ef30696',0,1,'reading','my book reading activities','The Nightingale','Hitchhiker'),
 ('gpt4_ec93e27f',0,1,'travel','the transport I used','bus ride','train ride'),
 ('gpt4_f420262c',0,1,'travel','my airline travel','JetBlue','10,000 miles'),
 ('gpt4_74aed68e',0,1,'motoring','my car maintenance and racing activities','spark plugs','Turbocharged Tuesdays'),
 ('gpt4_21adecb5',0,1,'education','my academic milestones','undergraduate degree','master\'s thesis'),
 ('eac54adc',0,1,'work','my freelance business milestones','launched my website','signed a contract'),
 ('gpt4_68e94287',0,1,'social_media','my social media activities','vegan chili','PlankChallenge'),
 ('gpt4_e414231e',0,1,'cycling','my bicycle maintenance','flat tire','clipless pedals'),
 ('gpt4_2655b836',1,0,'motoring','the service and GPS repair of my car','GPS system','first time on March 15th'),
 ('gpt4_2487a7cb',0,1,'education','the webinar and workshop I attended','Data Analysis using Python','Effective Time Management'),
 ('gpt4_76048e76',1,0,'cycling','my vehicle care activities','gears were acting up','washed my current Corolla'),
 ('gpt4_2312f94c',0,1,'devices','my laptop and phone purchases','Dell XPS 13','Best Buy store'),
 ('2c63a862',1,0,'housing','my homebuying progress','house that I really love','started working with her'),
 ('2a1811e2',1,0,'community','my participation in religious events','Sunday mass','Holi'),
 ('bbf86515',0,1,'motoring','the racing events I attended','Turbocharged Tuesdays','Rack Fest'),
 ('gpt4_0b2f1d21',0,1,'kitchen','my kitchen appliance experiences','repair shop','bought it about three weeks ago'),
 ('f0853d11',1,0,'charity','the volunteering events I attended','Walk for Hunger','Coastal Cleanup'),
 ('gpt4_6ed717ea',1,0,'pets','the pet supplies I purchased','training pads','dog bed'),
 ('gpt4_2d58bcd6',1,0,'reading','the novels I finished','The Nightingale','The Hate U Give'),
 ('gpt4_65aabe59',1,2,'devices','the home technology I set up','home Wi-Fi router','smart thermostat'),
 ('b9cfe692',1,0,'reading','my reading experiences','Seven Husbands','The Nightingale'),
 ('gpt4_4edbafa2',1,0,'cooking','the BBQ gatherings I attended','3rd of June','17th of June'),
 ('c8090214',1,0,'shopping','my holiday shopping activities','Holiday Market','iPhone 13 Pro'),
 ('gpt4_483dd43c',1,0,'television','the television series I watched','Game of Thrones','The Crown'),
 ('e4e14d04',1,0,'reading','my participation in Book Lovers Unite','meetup','joined a Facebook group'),
 ('f9e8c073',0,1,'wellbeing','the bereavement support sessions I reported','three sessions','five sessions'),
 ('8fb83627',0,2,'reading','the National Geographic issues I had finished','third issue','fifth issue'),
 ('5831f84d',0,1,'education','the Crash Course videos I watched','10 videos','15 Crash Course videos'),
 ('e66b632c',0,1,'exercise','the personal best for a charity 5K that I reported','27 minutes and 45 seconds','26 minutes and 30 seconds'),
 ('cf22b7bf',0,1,'exercise','the weight loss I reported','5 pounds','10 pounds'),
 ('10e09553',0,1,'fishing','my fishing trips with Alex','7 largemouth bass','9 largemouth bass'),
 ('42ec0761',0,1,'devices','the availability of a screwdriver for my laptop','misplaced','spare screwdriver'),
 ('26bdc477',0,1,'photography','the number of trips I had taken my camera on','three trips','five trips'),
]

# 原文可在移除Date行后仍确定事件先后；只标明确日期或无重叠的相对时间。
# 值为阅读包中更早事件的证据索引，绝不沿用记录日期顺序。
EVENT_REVIEW = {
 'gpt4_2655b836': (0,'服务发生于3月15日，GPS问题发生于3月22日；记录顺序相反。'),
 'gpt4_2487a7cb': (0,'同日记录，两个月前的webinar早于上周六workshop。'),
 'gpt4_76048e76': (1,'二月中旬修车早于2月27日洗车。'),
 'gpt4_2312f94c': (1,'2月20日拿到手机，早于2月25日收到电脑；不能把1月预订当收到。'),
 '2c63a862': (0,'2月15日开始合作，早于3月1日看到房子；记录顺序相反。'),
 '2a1811e2': (0,'2月26日Holi早于3月19日mass；记录顺序相反。'),
 'bbf86515': (0,'6月14日赛事早于6月18日赛事。'),
 'gpt4_0b2f1d21': (0,'同在5月25日记录，上月送修早于约三周前购买咖啡机。'),
 'f0853d11': (1,'2月21日活动早于3月7日活动。'),
 'gpt4_2d58bcd6': (0,'在两周前读书会之前完成的书，早于上周末完成的书；记录顺序相反。'),
 'gpt4_4edbafa2': (1,'6月3日聚会早于6月17日聚会。'),
 'c8090214': (1,'Black Friday前一周活动早于Black Friday买手机。'),
 'e4e14d04': (0,'三周前加入早于上周参加meetup；记录顺序相反。'),
 '10e09553': (0,'7月10日捕鱼早于7月22日捕鱼，年份不改变同年先后关系。'),
}

EXCLUDED = {
 '0db4c65d':'书名原文与问题不一致，且讨论结束不等同读完。',
 'gpt4_4fc4f797':'第二条仍为次日计划，不能当作已发生测试。',
 'gpt4_f420262d':'与已选航空题近似同一事实背景，保守排除。',
 'gpt4_385a5000':'两条记录时间相同，不适用于主要记录顺序目标。',
 'gpt4_93159ced':'任职时长与职业时长是不同量，重复证据也不适合状态对照。',
 'gpt4_8c8961ae':'选定用户句没有明确另一趟旅行的证据。',
 'gpt4_fe651585':'两条用户句没有明确两次成为父母的先后。',
 '7e974930':'第一条在4月记录中罗列5月已完成销售，时间一致性不足。',
 '69fee5aa':'新记录只说新增一枚，未直接陈述当前总数；不把推断总数当普通事实标签。',
 '5c40ec5b':'第一条没有明确见面次数，普通事实对照不足。',
}


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def build():
    if (EXP/'protocol.json').exists():
        raise RuntimeError('确认性协议已冻结，禁止覆盖')
    review_path = ROOT/'data/processed/geometry_review_pool/review.json'
    pool = json.loads(review_path.read_text(encoding='utf-8'))
    indexed = {r['id']:r for r in pool}
    assert len(REVIEWED) == 50 and len({r[0] for r in REVIEWED}) == 50
    DATA.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    cases, documents, queries = [], [], []
    for qid, ai, bi, topic, subject, aval, bval in REVIEWED:
        source = indexed[qid]
        selected = [(index, source['evidence'][index], value) for index,value in ((ai,aval),(bi,bval))]
        selected.sort(key=lambda x:x[1]['date'])
        assert selected[0][1]['date'] < selected[1][1]['date'], qid
        assert selected[0][1]['session_fingerprint'] != selected[1][1]['session_fingerprint'], qid
        evidence = []
        for index, e, value in selected:
            user_text = e['text'].split('Assistant:',1)[0].strip()
            body = user_text.split('\n',1)[1]
            assert value.lower() in body.lower(), (qid,value)
            evidence.append({**{k:v for k,v in e.items() if k!='text'}, 'review_index':index,
                             'body':body, 'ordinary_value':value})
        assert all(e['ordinary_value'].lower() not in evidence[1-i]['body'].lower() for i,e in enumerate(evidence)), qid
        variants = {}
        for condition in ('dated','swapped_dates','no_date'):
            ids = []
            for i,e in enumerate(evidence):
                date = evidence[1-i]['date'] if condition == 'swapped_dates' else e['date']
                text = e['body'] if condition == 'no_date' else 'Date: '+date+'\n'+e['body']
                doc_id = 'gc_d_'+text_sha(qid+'\0'+condition+'\0'+text)[:24]
                documents.append({'id':doc_id,'text':text})
                ids.append(doc_id)
            variants[condition] = ids
        case = {'id':qid,'topic':topic,'evidence_group':source['evidence_group'], 'subject':subject,
                'source_type':source['question_type'], 'evidence':evidence, 'document_variants':variants,'queries':[],
                'record_order_label':'比较这两条记录的Date行，回答记录陈述顺序；不推断事件发生顺序。'}
        templates = (
            (f'Of the two messages about {subject}, retrieve the one posted first.', f'Of the two messages about {subject}, retrieve the one posted last.'),
            (f'Which dated account of {subject} comes before the other account?',f'Which dated account of {subject} comes after the other account?'),
            (f'Use the message timestamps to find the initial report concerning {subject}.',f'Use the message timestamps to find the final report concerning {subject}.'),
        )
        def add(style, family, index, text):
            query_id = 'gc_q_'+text_sha(qid+'\0'+style+'\0'+str(family)+'\0'+str(index))[:24]
            queries.append({'id':query_id,'text':INSTRUCTION+text})
            case['queries'].append({'id':query_id,'style':style,'family':family,'target_index':index,'text':text})
        for family,pair in enumerate(templates):
            for i,text in enumerate(pair):
                add('record',family,i,text)
        for i,e in enumerate(evidence):
            add('ordinary',0,i,f'Retrieve the statement mentioning {e["ordinary_value"]} in connection with {subject}.')
        add('neutral',0,0,f'What did I say about {subject}?')
        if qid in EVENT_REVIEW:
            earlier,reason = EVENT_REVIEW[qid]
            earlier_index = next(i for i,e in enumerate(evidence) if e['review_index']==earlier)
            case['event_review'] = {'earlier_index':earlier_index,'reason':reason,
                                    'label':'按用户正文叙述的事件先后，忽略消息Date行。'}
            event_pairs = (
                (f'From the two accounts of {subject}, retrieve the event that happened first, regardless of when it was reported.', f'From the two accounts of {subject}, retrieve the event that happened last, regardless of when it was reported.'),
                (f'For {subject}, which described occurrence preceded the other?', f'For {subject}, which described occurrence followed the other?'),
                (f'Use the timing stated inside the messages about {subject} to select the earlier event.', f'Use the timing stated inside the messages about {subject} to select the later event.'),
            )
            for family,pair in enumerate(event_pairs):
                for j,text in enumerate(pair):
                    add('event',family,earlier_index if j==0 else 1-earlier_index,text)
        cases.append(case)
    assert len({c['evidence_group'] for c in cases}) == 50, '共享证据组必须先合并'
    # 公开锚点只验证服务端与旧向量的兼容性，不参与本轮评分。
    anchors = []
    old_data = ROOT/'data/processed/temporal_readout'
    for kind in ('documents','queries'):
        rows = [json.loads(x) for x in (old_data/(kind+'.jsonl')).read_text(encoding='utf-8').splitlines() if x]
        for row in sorted(rows,key=lambda r:r['id'])[:4]:
            anchor_id = 'gc_anchor_'+row['id']
            (documents if kind=='documents' else queries).append({'id':anchor_id,'text':row['text']})
            anchors.append({'id':anchor_id,'old_id':row['id'],'kind':kind})
    save(DATA/'annotations.json',cases)
    write_jsonl(DATA/'documents.jsonl',documents)
    write_jsonl(DATA/'queries.jsonl',queries)
    audit = [{'id':r['id'],'decision':'selected' if r['id'] in {c['id'] for c in cases} else 'excluded' if r['id'] in EXCLUDED else 'reserve',
              'reason':EXCLUDED.get(r['id'],'正文已阅读；按预定证据对与主题标注' if r['id'] in {c['id'] for c in cases} else '未纳入本轮50链，无模型指标筛选')}
             for r in pool if r['question_type']!='multi-session']
    save(DATA/'review_audit.json',audit)
    manifest = {'source_sha256':sha(ROOT/'data/raw/longmemeval_s_cleaned.json'),'review_sha256':sha(review_path),
                'chains':50,'evidence_groups':50,'topic_clusters':len({c['topic'] for c in cases}),
                'event_chains':len(EVENT_REVIEW),'documents':len(documents),'queries':len(queries),'anchors':anchors,
                'annotation':'单一AI逐条阅读用户句；没有独立第二标注者；官方答案不直接作为新标签。',
                'limits':['来源仍是LongMemEval构造对话；新来源题/证据并不等于新主题或新数据集。',
                          '主要任务按记录时间选证据；事件顺序子任务单独标注和报告。',
                          'swap只改消息Date行，属于合成反事实，不改变事件正文；事件正文和消息日期可能不一致。',
                          'no_date移除消息元数据，正文日期保留；不宣称所有日期信息都被移除。',
                          'no_date记录顺序条件不赋主要正确率标签；只有事件子集正文顺序仍可识别。'],
                'input_sha256':{name:sha(DATA/name) for name in ('annotations.json','documents.jsonl','queries.jsonl','review_audit.json')}}
    save(DATA/'manifest.json',manifest)
    registry = json.loads((DEV/'selection_registry.json').read_text(encoding='utf-8'))
    checkpoints = {r['checkpoint']:sha(DEV/r['checkpoint']) for r in registry}
    if any(checkpoints[r['checkpoint']] != r['checkpoint_sha256'] for r in registry):
        raise ValueError('开发检查点已变动')
    primary_n = len({c['topic'] for c in cases})
    # 功效上限参考：最大方差的有界配对差值，50链不能保证识别小效应。
    power = {'note':'仅粗略正态近似，不是样本充分性保证；主题内完全相关时有效n更小',
             'chains':50,'topic_clusters':primary_n,
             'approx_80pct_minimum_detectable_difference_worst_variance':2.8/np.sqrt(primary_n)}
    save(EXP/'power_note.json',power)
    protocol = {'status':'编码和新测试评分前冻结', 'models':[r['name'] for r in registry[::3]],'seeds':[11,23,47],
                'primary':'dated记录顺序成对准确率：poincare_1减shared_euclidean；每链先平均三措辞/三种子，再按保守主题簇配对bootstrap。',
                'success':'主要差值95%区间下界>0且点估计至少5个百分点；三个种子差值方向均正；ordinary相对shared_euclidean非劣5个百分点。当前样本不足时明确报告不确定。',
                'ordinary_noninferiority':'dated普通事实单问：双曲减欧氏配对区间下界必须>-0.05；训练/验证上的门槛不能替代此项。',
                'secondary':['其他模型与学习曲率均探索性，不从测试挑赢家或曲率',
                             'swap反转记录顺序目标，ordinary目标不变；报告每链严格同时跟随反转比例',
                             'event子集正文顺序单独在dated/no_date/swapped_dates评估，目标始终按事件正文',
                             '保留双曲坐标仅换欧氏距离，检验曲率对预测及正确率的增量',
                             'neutral同问双目标的成对正确率必须0；日期解析规则仅作记录任务正对照'],
                'bootstrap':{'seed':20261016,'replicates':20000,'unit':'topic cluster','confidence':.95},
                'no_retraining':True,'old_test_reuse':False,'input_manifest_sha256':sha(DATA/'manifest.json'),
                'development_registry_sha256':sha(DEV/'selection_registry.json'),'checkpoints_sha256':checkpoints,
                'whitening_sha256':{name:sha(DEV/name) for name in ('whitening_mean.npy','whitening_std.npy')},
                'code_sha256':{name:sha(ROOT/'scripts'/name) for name in ('prepare_geometry_confirmatory.py','geometry_readouts.py')},
                'backend_anchor_gate':{'minimum_cosine':.999,'maximum_relative_l2':.02,'policy':'未通过则停止确认性评分，不能混用后端向量并归因于几何。'},
                'limitations':manifest['limits']}
    save(EXP/'protocol.json',protocol)
    config = yaml.safe_load((ROOT/'configs/temporal_fresh_test_encoding.yml').read_text(encoding='utf-8'))
    config['experiment'] = {'seed':20261016,'dataset_manifest':str((DATA/'manifest.json').relative_to(ROOT)),'output_dir':str(EMBED.relative_to(ROOT))}
    config['inputs'] = {k:str((DATA/(k+'.jsonl')).relative_to(ROOT)) for k in ('documents','queries')}
    config['embedding']['batch_size'] = 32
    for key in config['cache']:
        config['cache'][key] = str((EMBED/Path(config['cache'][key]).name).relative_to(ROOT))
    config_path = ROOT/'configs/geometry_confirmatory_encoding.yml'
    config_path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('chains','topic_clusters','event_chains','documents','queries')},ensure_ascii=False))


if __name__ == '__main__':
    build()
