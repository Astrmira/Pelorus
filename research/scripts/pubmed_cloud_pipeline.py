"""云端编码、训练与评分流水线；阶段与失败状态持久保存。"""

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,UTC

from prepare_pubmed_geometry import ROOT,DATA,EXP,sha,save


def status(stage,**extra):
    save(EXP/'pipeline_status.json',{'stage':stage,'updated_at_utc':datetime.now(UTC).isoformat(),'pid':os.getpid(),**extra})
    print('流水线阶段：'+stage,flush=True)


def main():
    if (EXP/'pipeline_status.json').exists():
        raise RuntimeError('流水线已有启动记录，禁止自动重复执行；失败须审计后恢复')
    lock=json.loads((EXP/'encoding_lock.json').read_text(encoding='utf-8'))
    if sha(DATA/'manifest.json')!=lock['manifest_sha256']:
        raise ValueError('数据清单与冻结记录不符')
    for name,expected in lock['code_sha256'].items():
        if sha(ROOT/'scripts'/name)!=expected:
            raise ValueError('冻结代码发生变化')
    if sha(ROOT/'configs/pubmed_geometry_encoding.yml')!=lock['config_sha256']:
        raise ValueError('编码配置发生变化')
    qa=json.loads((EXP/'data_review.json').read_text(encoding='utf-8'))
    if not qa['passed'] or qa['manifest_sha256']!=sha(DATA/'manifest.json'):
        raise ValueError('数据审计未通过或不对应冻结输入')
    try:
        status('编码中')
        subprocess.run([sys.executable,'-u','scripts/encode_embeddings_api.py','--config','configs/pubmed_geometry_encoding.yml'],cwd=ROOT,check=True)
        os.environ.pop('EMBEDDING_API_KEY',None)
        status('训练与验证选模中')
        subprocess.run([sys.executable,'-u','scripts/run_pubmed_geometry.py'],cwd=ROOT,check=True)
        result=json.loads((EXP/'results.json').read_text(encoding='utf-8'))
        if not result['verification']['all_checkpoint_hashes_match']:
            raise ValueError('完成后的模型检查点哈希不符')
        status('已完成',results_sha256=sha(EXP/'results.json'),report='report.md')
    except Exception as error:
        status('失败',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':
    main()
