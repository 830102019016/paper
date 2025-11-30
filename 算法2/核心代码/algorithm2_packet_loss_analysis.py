"""
Algorithm 2 - Packet Loss Analysis due to Handover
评估 Demand-Aware vs Greedy 在切换导致的丢包率上的对比

假设:
- 每次切换导致约 400ms 的中断
- 在此期间无法传输数据,产生丢包
- 丢包率 = (切换次数 × 切换延迟) / 总仿真时长 × 100%

作者: 复现实现
日期: 2025-11-26
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from datetime import datetime, timezone

# 复用 algorithm2_demand_aware.py 的核心模块
import sys
sys.path.insert(0, r'e:\uav\算法2\files 2')

from algorithm2_demand_aware import (
    LEOConstellation,
    GreedySelector,
    DemandAwareSelector,
    UAVLocation,
    TransmissionRecord,
    count_switches,
    run_simulation
)

# ============================================================================
# 丢包分析参数
# ============================================================================

class PacketLossParameters:
    """丢包分析参数"""

    HANDOVER_DELAY_MS = 400  # 切换延迟 (ms)
    SIMULATION_DURATION_MIN = 10  # 仿真时长 (分钟) - 与 algorithm2_demand_aware 保持一致

    @classmethod
    def calculate_packet_loss_rate(cls, num_handovers: int) -> float:
        """
        计算丢包率

        丢包率 = (切换次数 × 切换延迟) / 总时长

        Args:
            num_handovers: 切换次数

        Returns:
            丢包率 (百分比)
        """
        total_time_ms = cls.SIMULATION_DURATION_MIN * 60 * 1000  # 转为毫秒
        lost_time_ms = num_handovers * cls.HANDOVER_DELAY_MS
        packet_loss_rate = (lost_time_ms / total_time_ms) * 100
        return packet_loss_rate

    @classmethod
    def calculate_data_loss_mb(cls, num_handovers: int, avg_rate_mbps: float) -> float:
        """
        计算丢失的数据量

        Args:
            num_handovers: 切换次数
            avg_rate_mbps: 平均速率 (Mbps)

        Returns:
            丢失数据量 (MB)
        """
        lost_time_sec = num_handovers * (cls.HANDOVER_DELAY_MS / 1000)
        data_loss_mb = avg_rate_mbps * lost_time_sec / 8  # Mbps * 秒 / 8 = MB
        return data_loss_mb

# ============================================================================
# 丢包分析函数
# ============================================================================

def analyze_packet_loss(all_results: Dict[str, Dict[str, List[TransmissionRecord]]],
                       all_switch_counts: Dict[str, Dict[str, int]]) -> Dict:
    """
    分析丢包情况

    Args:
        all_results: 仿真结果
        all_switch_counts: 切换次数统计

    Returns:
        丢包分析结果
    """

    analysis = {
        'scenarios': [],
        'greedy': {
            'handovers': [],
            'packet_loss_rate': [],
            'data_loss_mb': [],
            'avg_rate_mbps': []
        },
        'demand_aware': {
            'handovers': [],
            'packet_loss_rate': [],
            'data_loss_mb': [],
            'avg_rate_mbps': []
        }
    }

    for scenario_key in all_results.keys():
        analysis['scenarios'].append(scenario_key)

        # Greedy 分析
        greedy_records = all_results[scenario_key]['Greedy']
        greedy_handovers = all_switch_counts[scenario_key]['Greedy']
        greedy_rates = [r.data_rate_mbps for r in greedy_records if r.data_rate_mbps > 0]
        greedy_avg_rate = np.mean(greedy_rates) if greedy_rates else 0

        greedy_loss_rate = PacketLossParameters.calculate_packet_loss_rate(greedy_handovers)
        greedy_data_loss = PacketLossParameters.calculate_data_loss_mb(greedy_handovers, greedy_avg_rate)

        analysis['greedy']['handovers'].append(greedy_handovers)
        analysis['greedy']['packet_loss_rate'].append(greedy_loss_rate)
        analysis['greedy']['data_loss_mb'].append(greedy_data_loss)
        analysis['greedy']['avg_rate_mbps'].append(greedy_avg_rate)

        # Demand-Aware 分析
        demand_records = all_results[scenario_key]['Demand-Aware']
        demand_handovers = all_switch_counts[scenario_key]['Demand-Aware']
        demand_rates = [r.data_rate_mbps for r in demand_records if r.data_rate_mbps > 0]
        demand_avg_rate = np.mean(demand_rates) if demand_rates else 0

        demand_loss_rate = PacketLossParameters.calculate_packet_loss_rate(demand_handovers)
        demand_data_loss = PacketLossParameters.calculate_data_loss_mb(demand_handovers, demand_avg_rate)

        analysis['demand_aware']['handovers'].append(demand_handovers)
        analysis['demand_aware']['packet_loss_rate'].append(demand_loss_rate)
        analysis['demand_aware']['data_loss_mb'].append(demand_data_loss)
        analysis['demand_aware']['avg_rate_mbps'].append(demand_avg_rate)

    return analysis

# ============================================================================
# 绘图函数
# ============================================================================

def plot_packet_loss_analysis(analysis: Dict):
    """绘制丢包分析图表 - 仅显示丢包率对比"""

    print("\n绘制丢包率对比图...")

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    scenarios = analysis['scenarios']
    x = np.arange(len(scenarios))
    width = 0.35

    colors = {
        'Greedy': '#E74C3C',
        'Demand-Aware': '#2ECC71'
    }

    # 丢包率对比柱状图
    bars1 = ax.bar(x - width/2, analysis['greedy']['packet_loss_rate'], width,
                   label='Greedy', color=colors['Greedy'], edgecolor='white')
    bars2 = ax.bar(x + width/2, analysis['demand_aware']['packet_loss_rate'], width,
                   label='Demand-Aware', color=colors['Demand-Aware'], edgecolor='white')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    ax.set_xlabel('Number of Satellites', fontsize=12, fontweight='bold')
    ax.set_ylabel('Packet Loss Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Packet Loss Rate Comparison (Handover Delay={PacketLossParameters.HANDOVER_DELAY_MS}ms)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    # 保存到 算法2\结果图表 目录
    import os
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '结果图表')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'packet_loss_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] 丢包分析图已保存: {output_file}")

    plt.show()

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""

    print("\n")
    print("+" + "="*58 + "+")
    print("|" + " "*10 + "丢包率分析 - Handover Impact" + " "*15 + "|")
    print("+" + "="*58 + "+")
    print(f"\n切换延迟假设: {PacketLossParameters.HANDOVER_DELAY_MS}ms")

    # Step 1: 准备TLE数据
    print("\n[Step 1] 准备TLE数据")
    tle_file = 'starlink.tle'

    import os
    if not os.path.exists(tle_file):
        print(f"未找到 {tle_file}")
        return
    else:
        print(f"[OK] TLE文件已存在: {tle_file}")

    # Step 2: 运行仿真
    print("\n[Step 2] 运行仿真 (3种卫星数量)")

    satellite_counts = [500, 1500, -1]  # 删除200卫星场景，与 algorithm2_demand_aware 保持一致
    demand_mbps = 8.0  # 1080p视频流需求

    all_results = {}
    all_switch_counts = {}

    start_time = datetime.now(timezone.utc)

    for num_sats in satellite_counts:
        try:
            results, switch_counts = run_simulation(
                tle_file,
                num_sats,
                demand_mbps=demand_mbps,
                duration_minutes=PacketLossParameters.SIMULATION_DURATION_MIN,
                time_step_sec=20,  # 从30s改为20s，与 algorithm2_demand_aware 保持一致
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

    # Step 3: 分析丢包
    print("\n[Step 3] 分析丢包情况")
    analysis = analyze_packet_loss(all_results, all_switch_counts)

    # 打印分析结果
    print("\n" + "="*70)
    print("丢包分析结果")
    print("="*70)
    for i, scenario in enumerate(analysis['scenarios']):
        print(f"\n{scenario} 卫星:")
        print(f"  Greedy:")
        print(f"    切换次数: {analysis['greedy']['handovers'][i]}")
        print(f"    丢包率: {analysis['greedy']['packet_loss_rate'][i]:.4f}%")
        print(f"    丢失数据: {analysis['greedy']['data_loss_mb'][i]:.2f} MB")
        print(f"  Demand-Aware:")
        print(f"    切换次数: {analysis['demand_aware']['handovers'][i]}")
        print(f"    丢包率: {analysis['demand_aware']['packet_loss_rate'][i]:.4f}%")
        print(f"    丢失数据: {analysis['demand_aware']['data_loss_mb'][i]:.2f} MB")

        # 计算改善
        improvement = ((analysis['greedy']['packet_loss_rate'][i] -
                       analysis['demand_aware']['packet_loss_rate'][i]) /
                      max(analysis['greedy']['packet_loss_rate'][i], 0.001)) * 100
        print(f"  改善: {improvement:.1f}%")

    # Step 4: 绘制图表
    print("\n[Step 4] 绘制丢包分析图表")
    try:
        plot_packet_loss_analysis(analysis)
    except Exception as e:
        print(f"[X] 绘图失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()
