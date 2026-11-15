"""云端层级意图编码、训练及有门槛的盲测流水线。"""

import json
import os
import subprocess
import sys
from datetime import datetime,UTC
from prepare_mesh_intent import ROOT,DATA,EXP,sha,save


def status(stage,**extra):
    save(EXP/'pipeline_status.json',{'stage':stage,'pid':os.getpid(),'utc':datetime.now(UTC).isoformat(),**extra})
    print('流水线阶段：'+stage,flush=True)


def main():
    if (EXP/'pipeline_status.json').exists():
        raise RuntimeError('已有启动记录，禁止自动重复执行')
    lock=json.loads((EXP/'encoding_lock.json').read_text(encoding='utf-8'))
    if sha(DATA/'manifest.json')!=lock['manifest_sha256'] or sha(ROOT/'configs/mesh_intent_encoding.yml')!=lock['config_sha256']:
        raise ValueError('冻结数据或编码配置不一致')
    for name,h in lock['code_sha256'].items():
        if sha(ROOT/'scripts'/name)!=h:
            raise ValueError('冻结代码不一致')
    review=json.loads((EXP/'data_review.json').read_text(encoding='utf-8'))
    if not review['passed'] or review['manifest_sha256']!=sha(DATA/'manifest.json'):
        raise ValueError('数据审计未通过')
    try:
        status('编码中')
        subprocess.run([sys.executable,'-u','scripts/encode_embeddings_api.py','--config','configs/mesh_intent_encoding.yml'],cwd=ROOT,check=True)
        os.environ.pop('EMBEDDING_API_KEY',None)
        status('训练与验证中，测试未评分')
        subprocess.run([sys.executable,'-u','scripts/run_mesh_intent.py'],cwd=ROOT,check=True)
        result=json.loads((EXP/'results.json').read_text(encoding='utf-8'))
        status(result['stage'],test_scored=result['test_scored'],results_sha256=sha(EXP/'results.json'))
    except Exception as error:
        status('失败',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':
    main()
