"""共享参数化的欧氏与标准Poincare读出；曲率为K=-c。"""

import torch
from torch import nn
from torch.nn import functional as F


def exp_origin(v, c):
    """使用正交切空间坐标，使原点测地距离等于输入向量范数。"""
    c = torch.as_tensor(c, dtype=v.dtype, device=v.device)
    if float(c.detach()) == 0:
        return v / 2
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t = c.sqrt() * norm / 2
    return (torch.tanh(t) / t) * v / 2


def poincare_squared(x, y, c):
    """标准球内精确测地距离平方；asinh形式避免acosh在重合点的奇异梯度。"""
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    delta2 = (x - y).square().sum(-1)
    denominator = ((1 - c * x.square().sum(-1)) * (1 - c * y.square().sum(-1))).clamp_min(1e-14)
    ratio = delta2 / denominator
    z = (c * ratio).clamp_min(1e-24).sqrt()
    return 4 * ratio * (torch.asinh(z) / z).square()


def tangent_distance_squared(x, y, c):
    return poincare_squared(exp_origin(x, c), exp_origin(y, c), c)


class GeometryReadout(nn.Module):
    def __init__(self, name, input_dim, output_dim=2):
        super().__init__()
        self.name = name
        if name == 'raw_cosine':
            return
        if name == 'bilinear':
            # 相同输出维度下双线性使用两倍参数，作为更强的不对称对照。
            self.query_map = nn.Linear(input_dim, output_dim, bias=False)
            self.document_map = nn.Linear(input_dim, output_dim, bias=False)
        elif name == 'nonlinear_euclidean':
            self.map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh(), nn.Linear(16, output_dim))
        else:
            self.map = nn.Linear(input_dim, output_dim, bias=False)
        if name == 'poincare_learned':
            self.curvature_logit = nn.Parameter(torch.tensor(-1.0986122886681098))

    def curvature(self):
        if self.name == 'poincare_learned':
            return 4 * self.curvature_logit.sigmoid()
        return {'poincare_01': 0.1, 'poincare_1': 1.0, 'poincare_4': 4.0}.get(self.name, 0.0)

    def coordinates(self, x):
        v = self.map(x)
        # 所有共享距离模型均采用同一个数值半径约束；6以内为恒等变换。
        return v * (6 / v.norm(dim=-1, keepdim=True).clamp_min(6))

    def forward(self, q, docs):
        if self.name == 'raw_cosine':
            return (F.normalize(q, dim=-1).unsqueeze(-2) * F.normalize(docs, dim=-1)).sum(-1)
        if self.name == 'bilinear':
            return (self.query_map(q).unsqueeze(-2) * self.document_map(docs)).sum(-1)
        qv, dv = self.coordinates(q).unsqueeze(-2), self.coordinates(docs)
        if self.name == 'shared_cosine':
            return (F.normalize(qv, dim=-1) * F.normalize(dv, dim=-1)).sum(-1)
        return -tangent_distance_squared(qv, dv, self.curvature())
