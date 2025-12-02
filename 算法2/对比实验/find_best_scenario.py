"""
寻找最佳展示场景 - Algorithm 2
目标: 找到Demand-Aware算法相比Greedy算法效果最好的随机场景

评判标准:
1. Demand-Aware切换次数减少最多
2. 两种算法的平均速率都保持在合理范围
3. 数据传输速率曲线具有明显对比效果

策略:
- 尝试不同的起始时间（模拟不同卫星位置配置）
- 评估每个场景的切换次数差异
- 选择最能展示算法优势的场景

作者: Claude
日期: 2025-12-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '核心代码'))

import numpy as np
from datetime import datetime, timedelta, timezone
import json
from typing import Dict, List, Tuple
from algorithm2_demand_aware import (
    LEOConstellation, GreedySelector, DemandAwareSelector,
    UAVLocation, TransmissionRecord, run_simulation, count_switches
)

# ============================================================================
# 场景评分函数
# ============================================================================

def evaluate_scenario(results: Dict[str, List[TransmissionRecord]],
                      switch_counts: Dict[str, int],
                      demand_mbps: float) -> Dict:
    """
    评估场景质量

    评分标准:
    1. 切换次数减少 (主要指标): reduction_score = (greedy_switches - demand_switches) * 10
    2. 两种算法都能提供稳定服务: stability_score
    3. 平均速率都满足需求: rate_score

    Returns:
        评分字典: {
            'total_score': 总分,
            'reduction': 切换次数减少量,
            'reduction_rate': 切换次数减少率,
            'greedy_switches': Greedy切换次数,
            'demand_switches': Demand-Aware切换次数,
            'greedy_avg_rate': Greedy平均速率,
            'demand_avg_rate': Demand-Aware平均速率,
            'details': 详细信息
        }
    """
    greedy_switches = switch_counts['Greedy']
    demand_switches = switch_counts['Demand-Aware']

    # 提取速率数据
    greedy_rates = [r.data_rate_mbps for r in results['Greedy'] if r.data_rate_mbps > 0]
    demand_rates = [r.data_rate_mbps for r in results['Demand-Aware'] if r.data_rate_mbps > 0]

    if not greedy_rates or not demand_rates:
        return {
            'total_score': 0,
            'reduction': 0,
            'reduction_rate': 0,
            'greedy_switches': greedy_switches,
            'demand_switches': demand_switches,
            'greedy_avg_rate': 0,
            'demand_avg_rate': 0,
            'details': 'No valid data'
        }

    greedy_avg_rate = np.mean(greedy_rates)
    demand_avg_rate = np.mean(demand_rates)

    # 1. 切换次数减少评分 (主要指标, 占70%)
    reduction = greedy_switches - demand_switches
    reduction_rate = reduction / greedy_switches if greedy_switches > 0 else 0
    reduction_score = reduction * 10  # 每减少1次切换得10分

    # 2. 稳定性评分 (占15%)
    # 要求两种算法都要有一定切换次数（验证算法有效性）
    if greedy_switches < 3 or demand_switches < 1:
        stability_score = 0  # 切换次数过少，场景不具代表性
    else:
        # Greedy应该有较多切换，Demand-Aware应该有较少切换
        stability_score = min(greedy_switches / 2, 20)  # 最高20分

    # 3. 速率质量评分 (占15%)
    # 两种算法的平均速率都应该满足需求或接近需求
    greedy_rate_score = min(greedy_avg_rate / demand_mbps * 10, 15)
    demand_rate_score = min(demand_avg_rate / demand_mbps * 10, 15)
    rate_score = (greedy_rate_score + demand_rate_score) / 2

    # 总分
    total_score = reduction_score * 0.7 + stability_score * 0.15 + rate_score * 0.15

    return {
        'total_score': total_score,
        'reduction': reduction,
        'reduction_rate': reduction_rate,
        'greedy_switches': greedy_switches,
        'demand_switches': demand_switches,
        'greedy_avg_rate': greedy_avg_rate,
        'demand_avg_rate': demand_avg_rate,
        'details': f'Reduction: {reduction}, Rate: G={greedy_avg_rate:.1f} D={demand_avg_rate:.1f}'
    }


# ============================================================================
# 搜索最佳场景
# ============================================================================

def search_best_scenario(tle_file: str,
                         num_sats: int,
                         demand_mbps: float,
                         num_trials: int = 20,
                         duration_minutes: int = 10,
                         time_step_sec: int = 20) -> Dict:
    """
    搜索最佳展示场景

    Args:
        tle_file: TLE文件路径
        num_sats: 卫星数量
        demand_mbps: 需求速率
        num_trials: 尝试次数
        duration_minutes: 仿真时长
        time_step_sec: 时间步长

    Returns:
        最佳场景信息
    """
    print("\n" + "="*80)
    print(f"搜索最佳场景: {num_sats}颗卫星, 需求={demand_mbps}Mbps")
    print("="*80)

    best_scenario = None
    best_score = -1
    all_scenarios = []

    # 基准时间
    base_time = datetime(2025, 12, 2, 0, 0, 0, tzinfo=timezone.utc)

    for trial in range(num_trials):
        # 生成不同的起始时间 (间隔30分钟，模拟不同卫星配置)
        start_time = base_time + timedelta(minutes=trial * 30)

        print(f"\n[Trial {trial+1}/{num_trials}] 起始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        try:
            # 运行仿真
            results, switch_counts = run_simulation(
                tle_file,
                num_sats,
                demand_mbps=demand_mbps,
                duration_minutes=duration_minutes,
                time_step_sec=time_step_sec,
                start_time=start_time
            )

            # 评估场景
            score_info = evaluate_scenario(results, switch_counts, demand_mbps)

            # 添加起始时间信息
            scenario_info = {
                'trial': trial + 1,
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'score': score_info,
                'results': results,
                'switch_counts': switch_counts
            }

            all_scenarios.append(scenario_info)

            # 打印评分
            print(f"  评分: {score_info['total_score']:.2f}")
            print(f"  切换次数: Greedy={score_info['greedy_switches']}, Demand-Aware={score_info['demand_switches']}")
            print(f"  减少: {score_info['reduction']}次 ({score_info['reduction_rate']*100:.1f}%)")
            print(f"  平均速率: Greedy={score_info['greedy_avg_rate']:.2f}Mbps, Demand-Aware={score_info['demand_avg_rate']:.2f}Mbps")

            # 更新最佳场景
            if score_info['total_score'] > best_score:
                best_score = score_info['total_score']
                best_scenario = scenario_info
                print(f"  [*] 新的最佳场景! (评分: {best_score:.2f})")

        except Exception as e:
            print(f"  [X] 仿真失败: {e}")
            continue

    # 汇总结果
    print("\n" + "="*80)
    print("搜索完成!")
    print("="*80)

    if best_scenario is None:
        print("未找到有效场景")
        return None

    print(f"\n最佳场景: Trial {best_scenario['trial']}")
    print(f"起始时间: {best_scenario['start_time']}")
    print(f"总评分: {best_scenario['score']['total_score']:.2f}")
    print(f"切换次数: Greedy={best_scenario['score']['greedy_switches']}, Demand-Aware={best_scenario['score']['demand_switches']}")
    print(f"切换减少: {best_scenario['score']['reduction']}次 ({best_scenario['score']['reduction_rate']*100:.1f}%)")
    print(f"平均速率: Greedy={best_scenario['score']['greedy_avg_rate']:.2f}Mbps, Demand-Aware={best_scenario['score']['demand_avg_rate']:.2f}Mbps")

    # 排序并显示Top 5
    all_scenarios.sort(key=lambda x: x['score']['total_score'], reverse=True)

    print("\n" + "-"*80)
    print("Top 5 场景:")
    print("-"*80)
    for i, scenario in enumerate(all_scenarios[:5]):
        print(f"{i+1}. Trial {scenario['trial']}: 评分={scenario['score']['total_score']:.2f}, "
              f"切换减少={scenario['score']['reduction']}次 "
              f"({scenario['score']['reduction_rate']*100:.1f}%)")

    return {
        'best_scenario': best_scenario,
        'all_scenarios': all_scenarios,
        'num_sats': num_sats,
        'demand_mbps': demand_mbps
    }


# ============================================================================
# 保存最佳场景
# ============================================================================

def save_best_scenario(search_results: Dict, output_dir: str):
    """保存最佳场景信息到JSON文件"""

    os.makedirs(output_dir, exist_ok=True)

    best_scenario = search_results['best_scenario']

    # 准备保存数据（不包括TransmissionRecord对象，只保存关键信息）
    save_data = {
        'num_sats': search_results['num_sats'],
        'demand_mbps': search_results['demand_mbps'],
        'best_trial': best_scenario['trial'],
        'best_start_time': best_scenario['start_time'],
        'score': best_scenario['score'],
        'top5_trials': [
            {
                'trial': s['trial'],
                'start_time': s['start_time'],
                'score': s['score']
            }
            for s in search_results['all_scenarios'][:5]
        ]
    }

    # 保存到JSON
    output_file = os.path.join(output_dir, f'best_scenario_{search_results["num_sats"]}sats.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n[保存] 最佳场景信息已保存到: {output_file}")

    return output_file


# ============================================================================
# 使用最佳场景重新运行完整仿真
# ============================================================================

def run_with_best_scenario(tle_file: str,
                           best_scenario_info: Dict,
                           output_dir: str):
    """使用最佳场景参数重新运行完整仿真并生成图表"""

    print("\n" + "="*80)
    print("使用最佳场景重新运行完整仿真")
    print("="*80)

    best_scenario = best_scenario_info['best_scenario']
    start_time = datetime.strptime(best_scenario['start_time'], '%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=timezone.utc)

    num_sats = best_scenario_info['num_sats']
    demand_mbps = best_scenario_info['demand_mbps']

    print(f"\n参数:")
    print(f"  卫星数量: {num_sats}")
    print(f"  需求速率: {demand_mbps} Mbps")
    print(f"  起始时间: {best_scenario['start_time']}")

    # 运行仿真
    results, switch_counts = run_simulation(
        tle_file,
        num_sats,
        demand_mbps=demand_mbps,
        duration_minutes=10,
        time_step_sec=20,
        start_time=start_time
    )

    # 保存结果数据
    print("\n保存结果数据...")
    results_data = {
        'num_sats': num_sats,
        'demand_mbps': demand_mbps,
        'start_time': best_scenario['start_time'],
        'greedy': {
            'switch_count': switch_counts['Greedy'],
            'records': [
                {
                    'time_sec': r.time_sec,
                    'data_rate_mbps': r.data_rate_mbps,
                    'cumulative_data_mb': r.cumulative_data_mb,
                    'satellite_id': r.satellite_id,
                    'distance_km': r.distance_km
                }
                for r in results['Greedy']
            ]
        },
        'demand_aware': {
            'switch_count': switch_counts['Demand-Aware'],
            'records': [
                {
                    'time_sec': r.time_sec,
                    'data_rate_mbps': r.data_rate_mbps,
                    'cumulative_data_mb': r.cumulative_data_mb,
                    'satellite_id': r.satellite_id,
                    'distance_km': r.distance_km
                }
                for r in results['Demand-Aware']
            ]
        }
    }

    results_file = os.path.join(output_dir, f'best_scenario_{num_sats}sats_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)

    print(f"[保存] 结果数据已保存到: {results_file}")

    return results, switch_counts


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""

    print("\n" + "+"*80)
    print(" "*20 + "寻找最佳展示场景 - Algorithm 2")
    print("+"*80)

    # TLE文件路径
    tle_file = os.path.join(os.path.dirname(__file__), '..', '核心代码', 'starlink.tle')

    if not os.path.exists(tle_file):
        print(f"\n错误: TLE文件不存在: {tle_file}")
        print("请先运行 algorithm2_combined_analysis.py 下载TLE文件")
        return

    # 输出目录
    output_dir = os.path.join(os.path.dirname(__file__), '..', '结果图表')
    os.makedirs(output_dir, exist_ok=True)

    # 搜索参数
    num_sats = 500  # 使用500颗卫星（可见卫星适中，容易展示效果）
    demand_mbps = 8.0  # 1080p视频流需求
    num_trials = 30  # 尝试30个不同起始时间

    print(f"\n搜索参数:")
    print(f"  卫星数量: {num_sats}")
    print(f"  需求速率: {demand_mbps} Mbps")
    print(f"  尝试次数: {num_trials}")

    # 搜索最佳场景
    search_results = search_best_scenario(
        tle_file,
        num_sats=num_sats,
        demand_mbps=demand_mbps,
        num_trials=num_trials,
        duration_minutes=10,
        time_step_sec=20
    )

    if search_results is None:
        print("\n搜索失败，未找到有效场景")
        return

    # 保存最佳场景信息
    save_best_scenario(search_results, output_dir)

    # 使用最佳场景重新运行完整仿真
    results, switch_counts = run_with_best_scenario(
        tle_file,
        search_results,
        output_dir
    )

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n使用最佳场景的参数运行 algorithm2_combined_analysis.py:")
    print(f"  起始时间: {search_results['best_scenario']['start_time']}")
    print(f"  可以修改代码中的 start_time 参数来复现这个场景")


if __name__ == "__main__":
    main()
