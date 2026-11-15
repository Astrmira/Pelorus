"""使用当前后端重编码后的冻结模型，对相同新测试数据执行一次评估。"""

import evaluate_geometry_confirmatory as evaluation
from reset_geometry_backend import CURRENT_DEV, CURRENT_EXP, CURRENT_TEST_EMBED


if __name__ == '__main__':
    evaluation.DEV = CURRENT_DEV
    evaluation.EXP = CURRENT_EXP
    evaluation.EMBED = CURRENT_TEST_EMBED
    evaluation.main()
