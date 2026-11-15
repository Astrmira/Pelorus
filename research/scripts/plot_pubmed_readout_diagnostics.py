"""将已完成的验证集诊断绘制为独立科研图，不训练或选择模型。"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'artifacts/experiments/pubmed_readout_diagnostics_v1'


def main():
    result=json.loads((EXP/'results.json').read_text(encoding='utf-8'))
    if result['stage']!='已完成':
        raise ValueError('诊断尚未完成')
    font_manager.findfont('Microsoft YaHei',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'svg.fonttype':'path'})
    index={(r['name'],r['dimension']):r for r in result['summary']}
    dims=(8,32,128,256)
    reference=index['raw_cosine',1536]['validation']['full']['macro_r1']*100
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.6),sharey=True)
    series=((('random_cosine','正交随机投影','#7b8794'),('pca_cosine','训练文档PCA','#176bba')),
            (('random_local','随机初始化＋10候选训练','#2c9474'),('random_full','随机初始化＋全库训练','#dc8d28'),('pca_full','PCA初始化＋全库训练','#8255ac')))
    for ax,items,title in zip(axes,series,('未训练的压缩表示','按全库验证集选择训练检查点')):
        for name,label,color in items:
            values=[index[name,d]['validation']['full']['macro_r1']*100 for d in dims]
            ax.plot(dims,values,marker='o',lw=2,label=label,color=color)
        ax.axhline(reference,color='#252b33',linestyle='--',lw=1.2,label=f'原始1536维cosine：{reference:.1f}%')
        ax.set_xscale('log',base=2);ax.set_xticks(dims,labels=[str(d) for d in dims]);ax.set_ylim(0,102)
        ax.set_xlabel('投影维度');ax.set_title(title,fontsize=12);ax.grid(alpha=.22)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right',fontsize=8.5,frameon=False)
    axes[0].set_ylabel('全库验证宏平均Top-1正确率（%）')
    fig.suptitle('PubMed向量读出退化诊断',fontsize=15,y=.99)
    fig.text(.5,.015,'仅验证集诊断，非盲测｜100个验证查询、10个规范分支｜随机投影与学习模型取3个种子均值',ha='center',fontsize=8.5,color='#555555')
    fig.tight_layout(rect=(0,.05,1,.96))
    for suffix in ('png','svg'):
        fig.savefig(EXP/f'diagnostic_curves.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('已生成验证集维度诊断图（PNG与SVG）。')


if __name__=='__main__':
    main()
