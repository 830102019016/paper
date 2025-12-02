"""
使用最佳场景运行仿真并生成图表 - Algorithm 2
基于之前找到的最佳场景参数来展示算法效果

场景信息:
- Trial 14 (2025-12-02 09:45:00 UTC)
- 切换次数: Greedy=7次, Demand-Aware=2次
- 减少率: 71.4%
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '核心代码'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timezone
from scipy.interpolate import make_interp_spline
from algorithm2_demand_aware import run_simulation

# 设置专业论文字体（SCI标准）
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['grid.linewidth'] = 0.8
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markersize'] = 8

def plot_best_scenario(results, switch_counts, demand_mbps, num_sats, save_path=None):
    """绘制最佳场景的对比图"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'Greedy': '#3498db',       # 蓝色
        'Demand-Aware': '#9b59b6'  # 紫色
    }

    # ============ 左图: 数据传输速率对比 ============
    ax1 = axes[0]

    # Greedy曲线
    greedy_records = results['Greedy']
    time_greedy = np.array([r.time_sec / 60 for r in greedy_records])
    rate_greedy = np.array([r.data_rate_mbps for r in greedy_records])

    # 平滑曲线
    if len(time_greedy) > 3:
        time_smooth = np.linspace(time_greedy.min(), time_greedy.max(), 300)
        spl = make_interp_spline(time_greedy, rate_greedy, k=3)
        rate_smooth = spl(time_smooth)
        ax1.plot(time_smooth, rate_smooth,
                linestyle='-', color=colors['Greedy'], linewidth=2.5,
                label='Greedy', alpha=0.85)
    else:
        ax1.plot(time_greedy, rate_greedy,
                linestyle='-', marker='o',
                color=colors['Greedy'], linewidth=2.5, markersize=8,
                label='Greedy', alpha=0.85)

    # 标记Greedy切换点
    switch_times_greedy = []
    switch_rates_greedy = []
    prev_sat_id = None
    prev_time = None
    prev_rate = None
    for r in greedy_records:
        if prev_sat_id is not None and r.satellite_id != prev_sat_id and r.satellite_id != -1:
            if prev_time is not None and prev_rate is not None:
                switch_times_greedy.append(prev_time / 60)
                switch_rates_greedy.append(prev_rate)
        prev_sat_id = r.satellite_id
        prev_time = r.time_sec
        prev_rate = r.data_rate_mbps

    if switch_times_greedy:
        ax1.scatter(switch_times_greedy, switch_rates_greedy, marker='v', s=100,
                   color=colors['Greedy'], edgecolors='white',
                   linewidths=1.5, zorder=5, alpha=0.8,
                   label=f'Greedy Handovers ({len(switch_times_greedy)})')

    # Demand-Aware曲线
    demand_records = results['Demand-Aware']
    time_demand = np.array([r.time_sec / 60 for r in demand_records])
    rate_demand = np.array([r.data_rate_mbps for r in demand_records])

    if len(time_demand) > 3:
        time_smooth = np.linspace(time_demand.min(), time_demand.max(), 300)
        spl = make_interp_spline(time_demand, rate_demand, k=3)
        rate_smooth = spl(time_smooth)
        ax1.plot(time_smooth, rate_smooth,
                linestyle='-', color=colors['Demand-Aware'], linewidth=2.5,
                label='Demand-Aware', alpha=0.85)
    else:
        ax1.plot(time_demand, rate_demand,
                linestyle='-', marker='s',
                color=colors['Demand-Aware'], linewidth=2.5, markersize=8,
                label='Demand-Aware', alpha=0.85)

    # 标记Demand-Aware切换点
    switch_times_demand = []
    switch_rates_demand = []
    prev_sat_id = None
    prev_time = None
    prev_rate = None
    for r in demand_records:
        if prev_sat_id is not None and r.satellite_id != prev_sat_id and r.satellite_id != -1:
            if prev_time is not None and prev_rate is not None:
                switch_times_demand.append(prev_time / 60)
                switch_rates_demand.append(prev_rate)
        prev_sat_id = r.satellite_id
        prev_time = r.time_sec
        prev_rate = r.data_rate_mbps

    if switch_times_demand:
        ax1.scatter(switch_times_demand, switch_rates_demand, marker='s', s=100,
                   color=colors['Demand-Aware'], edgecolors='white',
                   linewidths=1.5, zorder=5, alpha=0.8,
                   label=f'Demand-Aware Handovers ({len(switch_times_demand)})')

    # 添加需求速率基准线
    ax1.axhline(y=demand_mbps, color='red', linestyle='--', linewidth=2,
                label=f'Demand ({demand_mbps} Mbps)', alpha=0.7)

    ax1.set_ylabel('Data Rate (Mbps)', fontsize=13)
    ax1.set_xlabel('Time (min)\n(a) Data Rate Comparison', fontsize=13)
    ax1.legend(fontsize=9, framealpha=0.95, edgecolor='black',
              fancybox=False, shadow=False, loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.tick_params(direction='in', which='both')

    # ============ 右图: 切换次数柱状图 ============
    ax2 = axes[1]

    methods = ['Greedy', 'Demand-Aware']
    handover_counts = [switch_counts['Greedy'], switch_counts['Demand-Aware']]

    x = np.arange(len(methods))
    bars = ax2.bar(x, handover_counts, width=0.6,
                   color=[colors['Greedy'], colors['Demand-Aware']],
                   edgecolor='black', linewidth=1.2, alpha=0.9)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', fontsize=12, fontweight='bold')

    # 计算减少百分比
    reduction = switch_counts['Greedy'] - switch_counts['Demand-Aware']
    reduction_rate = reduction / switch_counts['Greedy'] * 100

    ax2.set_ylabel('Number of Handovers', fontsize=13)
    ax2.set_xlabel(f'Method\n(b) Handover Count\nReduction: {reduction} ({reduction_rate:.1f}%)', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.8)
    ax2.tick_params(direction='in', which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[保存] 图表已保存到: {save_path}")

    plt.show()


def main():
    """主程序"""

    print("\n" + "="*80)
    print("使用最佳场景运行仿真 - Algorithm 2")
    print("="*80)

    # 最佳场景参数（从搜索结果中获得）
    num_sats = 500
    demand_mbps = 8.0
    start_time = datetime(2025, 12, 2, 9, 45, 0, tzinfo=timezone.utc)  # Trial 14

    print(f"\n场景参数:")
    print(f"  卫星数量: {num_sats}")
    print(f"  需求速率: {demand_mbps} Mbps")
    print(f"  起始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  预期效果: Greedy=7次切换, Demand-Aware=2次切换, 减少71.4%")

    # TLE文件
    tle_file = os.path.join(os.path.dirname(__file__), '..', '核心代码', 'starlink.tle')

    if not os.path.exists(tle_file):
        print(f"\n错误: TLE文件不存在: {tle_file}")
        return

    # 运行仿真
    print("\n运行仿真...")
    results, switch_counts = run_simulation(
        tle_file,
        num_sats,
        demand_mbps=demand_mbps,
        duration_minutes=10,
        time_step_sec=20,
        start_time=start_time
    )

    # 输出结果
    print("\n" + "="*80)
    print("仿真结果:")
    print("="*80)
    print(f"Greedy算法:")
    print(f"  切换次数: {switch_counts['Greedy']}")
    greedy_rates = [r.data_rate_mbps for r in results['Greedy'] if r.data_rate_mbps > 0]
    if greedy_rates:
        print(f"  平均速率: {np.mean(greedy_rates):.2f} Mbps")

    print(f"\nDemand-Aware算法:")
    print(f"  切换次数: {switch_counts['Demand-Aware']}")
    demand_rates = [r.data_rate_mbps for r in results['Demand-Aware'] if r.data_rate_mbps > 0]
    if demand_rates:
        print(f"  平均速率: {np.mean(demand_rates):.2f} Mbps")

    reduction = switch_counts['Greedy'] - switch_counts['Demand-Aware']
    reduction_rate = reduction / switch_counts['Greedy'] * 100 if switch_counts['Greedy'] > 0 else 0

    print(f"\n切换次数减少: {reduction}次 ({reduction_rate:.1f}%)")

    # 生成图表
    output_dir = os.path.join(os.path.dirname(__file__), '..', '结果图表')
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, 'best_scenario_comparison.png')

    print("\n生成图表...")
    plot_best_scenario(results, switch_counts, demand_mbps, num_sats, save_path)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)


if __name__ == "__main__":
    main()
