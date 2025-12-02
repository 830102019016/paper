"""
Algorithm 2 - Combined Analysis (Demand-Aware + Packet Loss)
合并分析：需求感知策略 + 丢包率分析

包含两个独立的图表：
1. 图1 (2x1): 折线图 - 卫星数量对比
   - (a) 累计数据传输量
   - (b) 平均数据传输速率

2. 图2 (单图): 柱状图 - 丢包率对比 (Greedy vs Demand-Aware)

作者: 复现实现
日期: 2025-12-02
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from datetime import datetime, timezone
import os

# 复用 algorithm2_demand_aware.py 的核心模块
import sys
sys.path.insert(0, r'e:\paper\算法2\核心代码')

from algorithm2_demand_aware import (
    LEOConstellation,
    GreedySelector,
    DemandAwareSelector,
    UAVLocation,
    TransmissionRecord,
    count_switches,
    run_simulation
)

# 设置专业论文字体（SCI标准）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式使用STIX字体
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8

# ============================================================================
# 丢包分析参数
# ============================================================================

class PacketLossParameters:
    """丢包分析参数"""

    HANDOVER_DELAY_MS = 400  # 切换延迟 (ms)
    SIMULATION_DURATION_MIN = 10  # 仿真时长 (分钟)

    @classmethod
    def calculate_packet_loss_rate(cls, num_handovers: int) -> float:
        """
        计算丢包率
        丢包率 = (切换次数 × 切换延迟) / 总时长
        """
        total_time_ms = cls.SIMULATION_DURATION_MIN * 60 * 1000  # 转为毫秒
        lost_time_ms = num_handovers * cls.HANDOVER_DELAY_MS
        packet_loss_rate = (lost_time_ms / total_time_ms) * 100
        return packet_loss_rate

# ============================================================================
# 图1: 折线图 - 数据传输速率对比 (1x2布局)
# ============================================================================

def plot_line_charts(all_results: Dict[str, Dict[str, List[TransmissionRecord]]],
                     demand_mbps: float = 8.0,
                     save_path: str = None):
    """
    绘制折线图对比 (1行2列布局)
    左图: Greedy 在三个场景下的速率变化
    右图: Demand-Aware 在三个场景下的速率变化
    """
    from scipy.interpolate import make_interp_spline

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scenarios = list(all_results.keys())

    # 配色方案：不同卫星数量
    scenario_colors = {
        '500': '#3498db',    # 蓝色
        '1500': '#9b59b6',   # 紫色
        'All': '#2ecc71'     # 绿色
    }

    markers = {'500': 'o', '1500': 's', 'All': '^'}

    # ============ 子图(a): Greedy 算法 ============
    ax1 = axes[0]

    for scenario in scenarios:
        greedy_records = all_results[scenario]['Greedy']
        time_greedy = np.array([r.time_sec / 60 for r in greedy_records])  # 转为分钟
        rate_greedy = np.array([r.data_rate_mbps for r in greedy_records])

        color = scenario_colors.get(scenario, '#95a5a6')
        marker = markers.get(scenario, 'o')

        # 平滑曲线
        if len(time_greedy) > 3:
            time_smooth = np.linspace(time_greedy.min(), time_greedy.max(), 300)
            spl = make_interp_spline(time_greedy, rate_greedy, k=3)
            rate_smooth = spl(time_smooth)
            ax1.plot(time_smooth, rate_smooth,
                    linestyle='-', color=color, linewidth=2.5,
                    label=f'{scenario} sats', alpha=0.85)
        else:
            ax1.plot(time_greedy, rate_greedy,
                    linestyle='-', marker=marker,
                    color=color, linewidth=2.5, markersize=8,
                    label=f'{scenario} sats', alpha=0.85)

        # 标记切换点（标记切换前的最后一个点）
        switch_times = []
        switch_rates = []
        prev_sat_id = None
        prev_time = None
        prev_rate = None
        for r in greedy_records:
            if prev_sat_id is not None and r.satellite_id != prev_sat_id and r.satellite_id != -1:
                # 标记切换前的时间和速率（旧卫星的最后一个点）
                if prev_time is not None and prev_rate is not None:
                    switch_times.append(prev_time / 60)
                    switch_rates.append(prev_rate)
            prev_sat_id = r.satellite_id
            prev_time = r.time_sec
            prev_rate = r.data_rate_mbps

        if switch_times:
            ax1.scatter(switch_times, switch_rates, marker='v', s=80,
                       color=color, edgecolors='white',
                       linewidths=1.5, zorder=5, alpha=0.8)

    # 添加需求速率基准线
    ax1.axhline(y=demand_mbps, color='red', linestyle='--', linewidth=2,
                label=f'Demand ({demand_mbps} Mbps)', alpha=0.7)

    ax1.set_ylabel('Data Rate (Mbps)', fontsize=13)
    ax1.set_xlabel('Time (min)\n(a) Greedy', fontsize=13)
    ax1.legend(fontsize=10, framealpha=0.95, edgecolor='black',
              fancybox=False, shadow=False, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.tick_params(direction='in', which='both')

    # ============ 子图(b): Demand-Aware 算法 ============
    ax2 = axes[1]

    for scenario in scenarios:
        demand_records = all_results[scenario]['Demand-Aware']
        time_demand = np.array([r.time_sec / 60 for r in demand_records])
        rate_demand = np.array([r.data_rate_mbps for r in demand_records])

        color = scenario_colors.get(scenario, '#95a5a6')
        marker = markers.get(scenario, 'o')

        # 平滑曲线
        if len(time_demand) > 3:
            time_smooth = np.linspace(time_demand.min(), time_demand.max(), 300)
            spl = make_interp_spline(time_demand, rate_demand, k=3)
            rate_smooth = spl(time_smooth)
            ax2.plot(time_smooth, rate_smooth,
                    linestyle='-', color=color, linewidth=2.5,
                    label=f'{scenario} sats', alpha=0.85)
        else:
            ax2.plot(time_demand, rate_demand,
                    linestyle='-', marker=marker,
                    color=color, linewidth=2.5, markersize=8,
                    label=f'{scenario} sats', alpha=0.85)

        # 标记切换点（标记切换前的最后一个点）
        switch_times = []
        switch_rates = []
        prev_sat_id = None
        prev_time = None
        prev_rate = None
        for r in demand_records:
            if prev_sat_id is not None and r.satellite_id != prev_sat_id and r.satellite_id != -1:
                # 标记切换前的时间和速率（旧卫星的最后一个点）
                if prev_time is not None and prev_rate is not None:
                    switch_times.append(prev_time / 60)
                    switch_rates.append(prev_rate)
            prev_sat_id = r.satellite_id
            prev_time = r.time_sec
            prev_rate = r.data_rate_mbps

        if switch_times:
            ax2.scatter(switch_times, switch_rates, marker='s', s=80,
                       color=color, edgecolors='white',
                       linewidths=1.5, zorder=5, alpha=0.8)

    # 添加需求速率基准线
    ax2.axhline(y=demand_mbps, color='red', linestyle='--', linewidth=2,
                label=f'Demand ({demand_mbps} Mbps)', alpha=0.7)

    ax2.set_ylabel('Data Rate (Mbps)', fontsize=13)
    ax2.set_xlabel('Time (min)\n(b) Demand-Aware', fontsize=13)
    ax2.legend(fontsize=10, framealpha=0.95, edgecolor='black',
              fancybox=False, shadow=False, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.tick_params(direction='in', which='both')

    plt.tight_layout()

    if save_path:
        # Save PNG format
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 折线图已保存: {save_path}")

        # Save EPS format (vector, suitable for publication)
        eps_path = save_path.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"[OK] EPS格式已保存: {eps_path}")

    plt.show()

# ============================================================================
# 图2: 柱状图 - 切换次数和丢包率对比 (4条柱)
# ============================================================================

def plot_packet_loss_bar(all_results: Dict[str, Dict[str, List[TransmissionRecord]]],
                        all_switch_counts: Dict[str, Dict[str, int]],
                        save_path: str = None):
    """
    绘制柱状图对比
    4条柱: Greedy切换次数, Demand-Aware切换次数, Greedy丢包率, Demand-Aware丢包率
    蓝色: Greedy
    紫色: Demand-Aware
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scenarios = list(all_results.keys())
    x = np.arange(len(scenarios))
    width = 0.35

    # 配色方案：蓝色和紫色
    colors = {
        'Greedy': '#3498db',       # 蓝色
        'Demand-Aware': '#9b59b6'  # 紫色
    }

    # ============ 左图: 切换次数 ============
    ax1 = axes[0]

    greedy_handovers = [all_switch_counts[s]['Greedy'] for s in scenarios]
    demand_handovers = [all_switch_counts[s]['Demand-Aware'] for s in scenarios]

    bars1 = ax1.bar(x - width/2, greedy_handovers, width,
                    label='Greedy', color=colors['Greedy'],
                    edgecolor='black', linewidth=1.2, alpha=0.9)
    bars2 = ax1.bar(x + width/2, demand_handovers, width,
                    label='Demand-Aware', color=colors['Demand-Aware'],
                    edgecolor='black', linewidth=1.2, alpha=0.9)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', fontsize=10)

    ax1.set_ylabel('Number of Handovers', fontsize=13)
    ax1.set_xlabel('Number of Satellites\n(a) Handover Count', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend(fontsize=10, framealpha=0.95, edgecolor='black',
              fancybox=False, shadow=False, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.8)
    ax1.tick_params(direction='in', which='both')

    # ============ 右图: 丢包率 ============
    ax2 = axes[1]

    greedy_loss_rates = []
    demand_loss_rates = []

    for scenario in scenarios:
        greedy_handovers = all_switch_counts[scenario]['Greedy']
        demand_handovers = all_switch_counts[scenario]['Demand-Aware']

        greedy_loss = PacketLossParameters.calculate_packet_loss_rate(greedy_handovers)
        demand_loss = PacketLossParameters.calculate_packet_loss_rate(demand_handovers)

        greedy_loss_rates.append(greedy_loss)
        demand_loss_rates.append(demand_loss)

    bars3 = ax2.bar(x - width/2, greedy_loss_rates, width,
                    label='Greedy', color=colors['Greedy'],
                    edgecolor='black', linewidth=1.2, alpha=0.9)
    bars4 = ax2.bar(x + width/2, demand_loss_rates, width,
                    label='Demand-Aware', color=colors['Demand-Aware'],
                    edgecolor='black', linewidth=1.2, alpha=0.9)

    # 添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', fontsize=10)

    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', fontsize=10)

    ax2.set_ylabel('Packet Loss Rate (%)', fontsize=13)
    ax2.set_xlabel('Number of Satellites\n(b) Packet Loss Rate', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend(fontsize=10, framealpha=0.95, edgecolor='black',
              fancybox=False, shadow=False, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=0.8)
    ax2.tick_params(direction='in', which='both')

    plt.tight_layout()

    if save_path:
        # Save PNG format
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 柱状图已保存: {save_path}")

        # Save EPS format (vector, suitable for publication)
        eps_path = save_path.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"[OK] EPS格式已保存: {eps_path}")

    plt.show()

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""

    print("\n")
    print("+" + "="*70 + "+")
    print("|" + " "*15 + "Algorithm 2 - Combined Analysis" + " "*24 + "|")
    print("+" + "="*70 + "+")

    # Step 1: 准备TLE数据
    print("\n[Step 1] 准备TLE数据")
    tle_file = 'starlink.tle'

    if not os.path.exists(tle_file):
        print(f"[X] 未找到 {tle_file}")
        return
    else:
        print(f"[OK] TLE文件已存在: {tle_file}")

    # Step 2: 运行仿真
    print("\n[Step 2] 运行仿真 (3种卫星数量)")
    print("使用最佳场景参数: Trial 14 (2025-12-02 09:45:00 UTC)")

    satellite_counts = [500, 1500, -1]  # 500, 1500, All
    demand_mbps = 8.0  # 1080p视频流需求

    all_results = {}
    all_switch_counts = {}

    # 使用最佳场景的起始时间 (Trial 14)
    start_time = datetime(2025, 12, 2, 9, 45, 0, tzinfo=timezone.utc)

    for num_sats in satellite_counts:
        try:
            results, switch_counts = run_simulation(
                tle_file,
                num_sats,
                demand_mbps=demand_mbps,
                duration_minutes=PacketLossParameters.SIMULATION_DURATION_MIN,
                time_step_sec=20,
                start_time=start_time
            )
            label = str(num_sats) if num_sats != -1 else 'All'
            all_results[label] = results
            all_switch_counts[label] = switch_counts
        except Exception as e:
            print(f"[X] 仿真失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Step 3: 打印统计结果
    print("\n[Step 3] 统计结果")
    print("="*80)
    for scenario in all_results.keys():
        print(f"\n{scenario} 卫星:")
        print(f"  Greedy: 切换次数={all_switch_counts[scenario]['Greedy']}")
        print(f"  Demand-Aware: 切换次数={all_switch_counts[scenario]['Demand-Aware']}")

        greedy_loss = PacketLossParameters.calculate_packet_loss_rate(
            all_switch_counts[scenario]['Greedy'])
        demand_loss = PacketLossParameters.calculate_packet_loss_rate(
            all_switch_counts[scenario]['Demand-Aware'])

        improvement = ((greedy_loss - demand_loss) / max(greedy_loss, 0.001)) * 100

        print(f"  Greedy丢包率: {greedy_loss:.4f}%")
        print(f"  Demand-Aware丢包率: {demand_loss:.4f}%")
        print(f"  改善: {improvement:.1f}%")

    # Step 4: 生成图表
    print("\n[Step 4] 生成图表")

    output_dir = os.path.join(os.path.dirname(__file__), '..', '结果图表')
    os.makedirs(output_dir, exist_ok=True)

    # 图1: 折线图 (1x2)
    save_path_line = os.path.join(output_dir, 'demand_aware_number_of_satellites.png')
    plot_line_charts(all_results, demand_mbps=demand_mbps, save_path=save_path_line)

    # 图2: 柱状图
    save_path_bar = os.path.join(output_dir, 'packet_loss_analysis.png')
    plot_packet_loss_bar(all_results, all_switch_counts, save_path=save_path_bar)

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
