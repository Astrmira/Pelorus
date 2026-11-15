"""在当前后端全部重新训练冻结后，编码相同的新测试输入。"""

import json
import sys
from pathlib import Path
import encode_embeddings_api as api
from prepare_geometry_confirmatory import ROOT, DATA
from reset_geometry_backend import CURRENT_EXP, CURRENT_TEST_EMBED
from run_geometry_controlled import save, sha


def main():
    protocol = json.loads((CURRENT_EXP/'protocol.json').read_text(encoding='utf-8'))
    if sha(DATA/'manifest.json') != protocol['input_manifest_sha256']:
        raise ValueError('已冻结的新测试数据发生变化')
    evaluator = ROOT/'scripts/evaluate_geometry_confirmatory.py'
    config = ROOT/'configs/geometry_confirmatory_v2_encoding.yml'
    locks = {'protocol_sha256':sha(CURRENT_EXP/'protocol.json'),'evaluator_sha256':sha(evaluator),
             'encoder_sha256':sha(Path(__file__)),'config_sha256':sha(config)}
    lock = CURRENT_EXP/'evaluation_lock.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))!=locks:
        raise ValueError('编码前冻结的评估锁发生变化')
    save(lock,locks)
    sys.argv = [str(Path(api.__file__)),'--config',str(config)]
    api.main()
    save(CURRENT_EXP/'encoded_files_manifest.json', {name:sha(CURRENT_TEST_EMBED/name) for name in
                                                   ('documents.npy','queries.npy','document_ids.json','query_ids.json','runtime_manifest.json')})


if __name__ == '__main__':
    main()
