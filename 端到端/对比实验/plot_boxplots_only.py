"""
单独绘制数据分布箱线图
展示时延、能耗、成功率的统计分布特性

作者: 可视化脚本
日期: 2025-12-06
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 颜色配置
COLORS = {
    'G1': '#2ECC71',  # 绿色 - 提出方法
    'G2': '#3498DB',  # 蓝色
    'G3': '#F39C12',  # 橙色
    'G4': '#E74C3C'   # 红色 - 基线
}


def plot_distribution_boxplots(raw_results: dict, output_dir: str):
    """
    绘制能耗分布箱线图

    Args:
        raw_results: 原始实验结果
        output_dir: 输出目录
    """
    print("\n[绘制] 能耗分布箱线图...")

    fig, ax = plt.subplots(figsize=(10, 7))
    groups = ['G1', 'G2', 'G3', 'G4']

    # 提取能耗数据
    energies = {g: [r['e2e_metrics']['system_energy'] for r in raw_results[g]] for g in groups}

    # 绘制能耗箱线图
    bp = ax.boxplot([energies[g] for g in groups],
                    labels=[g for g in groups],
                    patch_artist=True,
                    notch=True,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=10))

    # 设置颜色
    for patch, g in zip(bp['boxes'], groups):
        patch.set_facecolor(COLORS[g])
        patch.set_alpha(0.7)

    # 美化箱线图
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], linewidth=1.5)

    # 设置标签和标题
    ax.set_ylabel('System Energy (J)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Group', fontsize=14, fontweight='bold')
    ax.set_title('System Energy Distribution Comparison',
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 添加统计信息（中位数标注）
    for i, g in enumerate(groups, 1):
        data = energies[g]
        median = np.median(data)
        ax.text(i, median, f'{median:.0f}J',
               ha='left', va='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    plt.tight_layout()

    # 保存
    output_file = os.path.join(output_dir, 'distribution_boxplots.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  [保存] {output_file}")

    plt.show()
    plt.close()

    # 打印统计信息
    print("\n  [能耗统计信息]")
    for g in groups:
        data = energies[g]
        print(f"  {g}: 中位数={np.median(data):.0f}J, "
              f"均值={np.mean(data):.0f}J, "
              f"标准差={np.std(data):.0f}J, "
              f"最小值={np.min(data):.0f}J, "
              f"最大值={np.max(data):.0f}J")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("绘制系统能耗分布箱线图")
    print("="*80)

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
        plot_distribution_boxplots(raw_results, output_dir)

        print("\n" + "="*80)
        print("✓ 能耗箱线图生成完成!")
        print(f"  输出目录: {output_dir}")
        print("\n生成的图表:")
        print("  - distribution_boxplots.png (系统能耗分布)")
        print("\n图表说明:")
        print("  - 箱体: 四分位数范围 (Q1-Q3)")
        print("  - notch: 中位数置信区间")
        print("  - 红色菱形: 均值")
        print("  - 横线: 中位数")
        print("  - 须线: 数据范围 (不含离群值)")
        print("  - 圆点: 离群值 (如有)")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n[错误] 生成图表失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
