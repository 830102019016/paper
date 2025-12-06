"""
单独绘制数据分布箱线图 (SCI论文标准)
展示系统能耗的统计分布特性 - G1 vs G4对比

作者: 可视化脚本
日期: 2025-12-06
版本: v2.0 (SCI优化版)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# SCI论文标准字体和样式设置
# ============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式使用STIX字体
plt.rcParams['axes.linewidth'] = 1.3
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 150

# ============================================================================
# 颜色配置 (SCI期刊友好配色 - 柔和专业色调)
# ============================================================================
# 使用柔和、专业的配色方案 (避免卡通化的饱和色)
COLORS = {
    'G1': '#4A90E2',  # 柔和蓝色 - 提出方法 (专业、信任感)
    'G4': '#E8A87C'   # 柔和橙色 - 传统基线 (温和对比)
}

# 备用配色 (如果期刊要求黑白打印)
COLORS_BW = {
    'G1': '#505050',  # 深灰
    'G4': '#A0A0A0'   # 浅灰
}


def plot_distribution_boxplots(raw_results: dict, output_dir: str, use_bw: bool = False):
    """
    绘制能耗分布箱线图 (SCI论文标准)
    仅对比G1 (Proposed) vs G4 (Baseline)

    Args:
        raw_results: 原始实验结果字典
        output_dir: 输出目录路径
        use_bw: 是否使用黑白配色 (默认False, 使用彩色)
    """
    print("\n[绘制] 系统能耗分布箱线图 (G1 vs G4, SCI标准)...")

    # 选择配色方案
    colors = COLORS_BW if use_bw else COLORS

    # 创建图形 (单列布局, 适合论文排版)
    fig, ax = plt.subplots(figsize=(8, 6))

    groups = ['G1', 'G4']
    group_labels = ['Adam-2opt +\nDemand-Aware', 'Basic-UAV +\nGreedy-LEO']

    # 提取能耗数据并转换为kJ (千焦耳)
    energies_j = {g: [r['e2e_metrics']['system_energy'] for r in raw_results[g]]
                  for g in groups}
    energies = {g: np.array(energies_j[g]) / 1000.0 for g in groups}  # 转换为kJ

    # 计算统计量 (用于后续标注)
    stats = {}
    for g in groups:
        data = np.array(energies[g])
        stats[g] = {
            'median': np.median(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75)
        }

    # 绘制标准矩形箱线图 (不使用notch, 避免蝴蝶形状)
    bp = ax.boxplot([energies[g] for g in groups],
                    labels=group_labels,
                    patch_artist=True,
                    notch=False,  # 不使用notch, 使用标准矩形箱体
                    showmeans=True,
                    widths=0.5,
                    meanprops=dict(
                        marker='D',
                        markerfacecolor='white',
                        markeredgecolor='#333333',
                        markersize=7,
                        markeredgewidth=1.0,
                        zorder=3
                    ),
                    medianprops=dict(
                        color='#333333',
                        linewidth=2.0,
                        zorder=3
                    ),
                    whiskerprops=dict(
                        color='#555555',
                        linewidth=1.0,
                        linestyle='-'
                    ),
                    capprops=dict(
                        color='#555555',
                        linewidth=1.0
                    ),
                    flierprops=dict(
                        marker='o',
                        markerfacecolor='#D32F2F',
                        markersize=5,
                        markeredgecolor='#333333',
                        markeredgewidth=0.5,
                        alpha=0.6
                    ))

    # 设置箱体颜色和样式 (半透明填充, 专业配色)
    for patch, g in zip(bp['boxes'], groups):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.5)  # 半透明 (0.4-0.6)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.0)

    # ========================================================================
    # 轴标签和网格 (SCI标准)
    # ========================================================================
    ax.set_ylabel('System Energy Consumption (kJ)',  # 使用kJ单位
                  fontsize=13, fontweight='normal')
    ax.set_xlabel('End-to-End Method Configuration',
                  fontsize=13, fontweight='normal')

    # 网格线 (仅Y轴, 虚线, 低透明度)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)  # 网格线置于底层

    # 刻度线方向 (向内, SCI期刊常见样式)
    ax.tick_params(direction='in', which='both', length=5, width=1.0)

    # Y轴范围优化 (留出适当空间)
    all_data = np.concatenate([energies[g] for g in groups])
    y_min = all_data.min() * 0.998
    y_max = all_data.max() * 1.008  # 留出更多顶部空间用于标注
    ax.set_ylim(y_min, y_max)

    # ========================================================================
    # 添加均值标注 (置于箱体正上方, 紧邻但不重叠)
    # ========================================================================
    for i, g in enumerate(groups, 1):
        mean_val = stats[g]['mean']
        q3_val = stats[g]['q3']
        # 标注位置: Q3正上方, 更小的偏移量 (紧贴箱体顶部)
        y_offset = (y_max - y_min) * 0.006  # 减小偏移量, 使标注更贴近箱体
        ax.text(i, q3_val + y_offset,
               f'{mean_val:.1f} kJ',  # 明确标注单位 "kJ"
               ha='center', va='bottom',
               fontsize=8, fontweight='normal',  # 更小字体 (9→8)
               color='#555555')  # 更柔和的灰色

    # ========================================================================
    # 计算统计信息
    # ========================================================================
    g1_median = stats['G1']['median']
    g4_median = stats['G4']['median']
    improvement = (g4_median - g1_median) / g4_median * 100
    absolute_saving = g4_median - g1_median

    # ========================================================================
    # 添加能耗降低标注 (右上角, 小号灰色文字, 低调专业)
    # ========================================================================
    ax.text(0.98, 0.97,
           f'Energy reduction: {improvement:.1f}%',
           transform=ax.transAxes,
           ha='right', va='top',
           fontsize=9,
           fontweight='normal',
           color='#666666',  # 柔和灰色
           bbox=dict(boxstyle='round,pad=0.4',
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.7))

    # ========================================================================
    # 保存图表 (多格式输出, SCI投稿标准)
    # ========================================================================
    plt.tight_layout()

    # 1. PNG格式 (高分辨率, 用于审阅)
    png_file = os.path.join(output_dir, 'energy_distribution_boxplot.png')
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  [已保存] PNG: {png_file}")

    # 2. EPS格式 (矢量图, SCI期刊要求)
    eps_file = os.path.join(output_dir, 'energy_distribution_boxplot.eps')
    plt.savefig(eps_file, format='eps', bbox_inches='tight')
    print(f"  [已保存] EPS: {eps_file}")

    # 3. PDF格式 (矢量图备用)
    pdf_file = os.path.join(output_dir, 'energy_distribution_boxplot.pdf')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
    print(f"  [已保存] PDF: {pdf_file}")

    plt.show()
    plt.close()

    # ========================================================================
    # 打印详细统计信息 (终端输出)
    # ========================================================================
    print("\n" + "="*70)
    print("  统计分析结果 (Statistical Analysis)")
    print("="*70)

    for g in groups:
        group_name = 'Proposed Framework (G1)' if g == 'G1' else 'Baseline Strategy (G4)'
        print(f"\n  {group_name}:")
        print(f"    Median (中位数)    : {stats[g]['median']:.2f} kJ  ({stats[g]['median']*1000:.0f} J)")
        print(f"    Mean (均值)        : {stats[g]['mean']:.2f} kJ  ({stats[g]['mean']*1000:.0f} J)")
        print(f"    Std Dev (标准差)   : {stats[g]['std']:.3f} kJ")
        print(f"    Q1 (下四分位数)    : {stats[g]['q1']:.2f} kJ")
        print(f"    Q3 (上四分位数)    : {stats[g]['q3']:.2f} kJ")
        print(f"    IQR (四分位距)     : {stats[g]['q3'] - stats[g]['q1']:.3f} kJ")
        print(f"    Sample Size (样本) : {len(energies[g])}")

    print("\n" + "-"*70)
    print("  性能改进 (Performance Improvement)")
    print("-"*70)
    print(f"    能耗降低百分比      : {improvement:.2f}%")
    print(f"    绝对节省能耗        : {absolute_saving:.2f} kJ  ({absolute_saving*1000:.1f} J)")
    print(f"    G1/G4 能耗比        : {g1_median/g4_median:.4f}")
    print("="*70 + "\n")

    # 打印推荐的图注文本
    print("\n" + "="*70)
    print("  推荐图注 (Recommended Figure Caption)")
    print("="*70)
    caption = f"""
Figure X. End-to-end system energy consumption under the proposed framework
and the baseline strategy. Each box summarizes {len(energies['G1'])} randomized
simulation runs with different IoT device deployment positions, with diamond
markers indicating the mean. The proposed method achieves a {improvement:.1f}%
reduction in total mission energy compared with the baseline.
"""
    print(caption)
    print("="*70 + "\n")

    # 返回统计信息供后续使用
    return stats, improvement


def main():
    """
    主函数 - 生成SCI论文标准的箱线图
    """
    print("\n" + "="*80)
    print("  系统能耗分布箱线图生成器 (SCI论文标准)")
    print("  End-to-End Energy Distribution Boxplot Generator")
    print("="*80)
    print("\n对比组别 (Comparison Groups):")
    print("  • G1 (Proposed Framework): P1 Optimization + Demand-Aware Strategy")
    print("  • G4 (Baseline Strategy): F-scheme + Greedy Strategy")
    print("\n输出格式 (Output Formats):")
    print("  • PNG (300 DPI) - 高分辨率位图")
    print("  • EPS - 矢量图 (SCI期刊标准)")
    print("  • PDF - 矢量图 (备用格式)")
    print("="*80 + "\n")

    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, '../结果图表')
    json_file = os.path.join(results_dir, 'e2e_experiment_results.json')

    # 加载结果
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        raw_results = data['raw_results']

        print(f"[加载] 结果文件: {json_file}")
        print(f"  实验日期: {data['metadata']['experiment_date']}")
        print(f"  数据点数量: {sum(len(raw_results[g]) for g in raw_results)}")

    except FileNotFoundError:
        print(f"\n[错误] 未找到结果文件: {json_file}")
        print("\n请先运行实验:")
        print("  cd ../核心代码")
        print("  python run_experiment.py")
        return
    except Exception as e:
        print(f"\n[错误] 加载结果失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建输出目录
    output_dir = results_dir
    os.makedirs(output_dir, exist_ok=True)

    # 生成图表
    try:
        stats, improvement = plot_distribution_boxplots(raw_results, output_dir)

        print("\n" + "="*80)
        print("  ✓ 箱线图生成完成! (Boxplot Generation Complete)")
        print("="*80)
        print(f"\n输出目录 (Output Directory):")
        print(f"  {output_dir}")
        print(f"\n生成的文件 (Generated Files):")
        print(f"  • energy_distribution_boxplot.png")
        print(f"  • energy_distribution_boxplot.eps  ← 推荐用于SCI投稿")
        print(f"  • energy_distribution_boxplot.pdf")
        print(f"\n图表元素说明 (Boxplot Elements):")
        print(f"  • 箱体 (Box)         : 标准矩形, Q1到Q3范围")
        print(f"  • 粗黑线 (Thick Line): 中位数 (Median)")
        print(f"  • 菱形 (Diamond)     : 均值 (Mean, 标注在箱体上方)")
        print(f"  • 须线 (Whiskers)    : 1.5×IQR范围")
        print(f"  • 圆点 (Circles)     : 离群值 (如有)")
        print(f"\n颜色方案 (Color Scheme - Muted Professional Tones):")
        print(f"  • 柔和蓝色 (Soft Blue)  : G1 Proposed Framework")
        print(f"  • 柔和橙色 (Soft Orange): G4 Baseline Strategy")
        print(f"  • 半透明填充 (alpha=0.5): 避免卡通化")
        print(f"\n设计改进 (Based on Reviewer Feedback v2):")
        print(f"  ✓ 标准矩形箱线图 (避免蝴蝶形状)")
        print(f"  ✓ 使用kJ单位 (提高可读性)")
        print(f"  ✓ 均值标注紧贴箱体顶部 (含单位 'kJ')")
        print(f"  ✓ 更小更柔和的标注字体 (8号, 灰色)")
        print(f"  ✓ 右上角低调显示能耗降低百分比")
        print(f"  ✓ 统一术语: Baseline Strategy (非Method)")
        print(f"  ✓ 柔和专业配色 (避免饱和色)")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n[错误] 生成图表失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
