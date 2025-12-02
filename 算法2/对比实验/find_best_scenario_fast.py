"""
快速寻找最佳展示场景 - Algorithm 2
优化版本: 更少的尝试次数，更快的执行速度
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '核心代码'))

import numpy as np
from datetime import datetime, timedelta, timezone
import json
from typing import Dict, List
from algorithm2_demand_aware import (
    LEOConstellation, GreedySelector, DemandAwareSelector,
    UAVLocation, TransmissionRecord, run_simulation
)

def evaluate_scenario(results: Dict[str, List[TransmissionRecord]],
                      switch_counts: Dict[str, int],
                      demand_mbps: float) -> Dict:
    """评估场景质量"""
    greedy_switches = switch_counts['Greedy']
    demand_switches = switch_counts['Demand-Aware']

    greedy_rates = [r.data_rate_mbps for r in results['Greedy'] if r.data_rate_mbps > 0]
    demand_rates = [r.data_rate_mbps for r in results['Demand-Aware'] if r.data_rate_mbps > 0]

    if not greedy_rates or not demand_rates:
        return {'total_score': 0, 'reduction': 0, 'reduction_rate': 0,
                'greedy_switches': greedy_switches, 'demand_switches': demand_switches,
                'greedy_avg_rate': 0, 'demand_avg_rate': 0}

    greedy_avg_rate = np.mean(greedy_rates)
    demand_avg_rate = np.mean(demand_rates)

    # 切换次数减少评分
    reduction = greedy_switches - demand_switches
    reduction_rate = reduction / greedy_switches if greedy_switches > 0 else 0
    reduction_score = reduction * 10

    # 稳定性评分
    if greedy_switches < 3 or demand_switches < 1:
        stability_score = 0
    else:
        stability_score = min(greedy_switches / 2, 20)

    # 速率质量评分
    greedy_rate_score = min(greedy_avg_rate / demand_mbps * 10, 15)
    demand_rate_score = min(demand_avg_rate / demand_mbps * 10, 15)
    rate_score = (greedy_rate_score + demand_rate_score) / 2

    total_score = reduction_score * 0.7 + stability_score * 0.15 + rate_score * 0.15

    return {
        'total_score': total_score,
        'reduction': reduction,
        'reduction_rate': reduction_rate,
        'greedy_switches': greedy_switches,
        'demand_switches': demand_switches,
        'greedy_avg_rate': greedy_avg_rate,
        'demand_avg_rate': demand_avg_rate
    }


def main():
    print("\n" + "="*80)
    print("快速搜索最佳展示场景 - Algorithm 2")
    print("="*80)

    # TLE文件
    tle_file = os.path.join(os.path.dirname(__file__), '..', '核心代码', 'starlink.tle')
    if not os.path.exists(tle_file):
        print(f"错误: TLE文件不存在: {tle_file}")
        return

    # 参数
    num_sats = 500
    demand_mbps = 8.0
    num_trials = 15  # 减少到15次
    duration_minutes = 10
    time_step_sec = 20

    print(f"\n参数: {num_sats}颗卫星, 需求={demand_mbps}Mbps, 尝试{num_trials}次\n")

    best_scenario = None
    best_score = -1
    all_results = []

    base_time = datetime(2025, 12, 2, 0, 0, 0, tzinfo=timezone.utc)

    for trial in range(num_trials):
        start_time = base_time + timedelta(minutes=trial * 45)  # 间隔45分钟

        print(f"[{trial+1}/{num_trials}] {start_time.strftime('%H:%M UTC')}...", end=' ', flush=True)

        try:
            results, switch_counts = run_simulation(
                tle_file, num_sats, demand_mbps=demand_mbps,
                duration_minutes=duration_minutes, time_step_sec=time_step_sec,
                start_time=start_time
            )

            score_info = evaluate_scenario(results, switch_counts, demand_mbps)

            scenario_data = {
                'trial': trial + 1,
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'score': score_info,
                'results': results,
                'switch_counts': switch_counts
            }

            all_results.append(scenario_data)

            print(f"评分={score_info['total_score']:.1f}, " +
                  f"切换: G={score_info['greedy_switches']} D={score_info['demand_switches']} " +
                  f"减少={score_info['reduction']}")

            if score_info['total_score'] > best_score:
                best_score = score_info['total_score']
                best_scenario = scenario_data
                print(f"     -> 新的最佳场景!")

        except Exception as e:
            print(f"失败: {e}")
            continue

    # 结果汇总
    print("\n" + "="*80)
    if best_scenario:
        print("最佳场景:")
        print(f"  Trial: {best_scenario['trial']}")
        print(f"  时间: {best_scenario['start_time']}")
        print(f"  评分: {best_scenario['score']['total_score']:.2f}")
        print(f"  切换: Greedy={best_scenario['score']['greedy_switches']}, "
              f"Demand-Aware={best_scenario['score']['demand_switches']}")
        print(f"  减少: {best_scenario['score']['reduction']}次 "
              f"({best_scenario['score']['reduction_rate']*100:.1f}%)")
        print(f"  速率: Greedy={best_scenario['score']['greedy_avg_rate']:.1f}Mbps, "
              f"Demand-Aware={best_scenario['score']['demand_avg_rate']:.1f}Mbps")

        # 保存结果
        output_dir = os.path.join(os.path.dirname(__file__), '..', '结果图表')
        os.makedirs(output_dir, exist_ok=True)

        # 保存场景信息
        save_data = {
            'num_sats': num_sats,
            'demand_mbps': demand_mbps,
            'best_trial': best_scenario['trial'],
            'best_start_time': best_scenario['start_time'],
            'score': best_scenario['score'],
            'top5': [
                {
                    'trial': s['trial'],
                    'start_time': s['start_time'],
                    'score': s['score']['total_score'],
                    'reduction': s['score']['reduction']
                }
                for s in sorted(all_results, key=lambda x: x['score']['total_score'], reverse=True)[:5]
            ]
        }

        output_file = os.path.join(output_dir, f'best_scenario_{num_sats}sats.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\n已保存到: {output_file}")

        # 保存完整结果数据
        results_data = {
            'num_sats': num_sats,
            'demand_mbps': demand_mbps,
            'start_time': best_scenario['start_time'],
            'greedy': {
                'switch_count': best_scenario['switch_counts']['Greedy'],
                'records': [
                    {
                        'time_sec': r.time_sec,
                        'data_rate_mbps': r.data_rate_mbps,
                        'satellite_id': r.satellite_id
                    }
                    for r in best_scenario['results']['Greedy']
                ]
            },
            'demand_aware': {
                'switch_count': best_scenario['switch_counts']['Demand-Aware'],
                'records': [
                    {
                        'time_sec': r.time_sec,
                        'data_rate_mbps': r.data_rate_mbps,
                        'satellite_id': r.satellite_id
                    }
                    for r in best_scenario['results']['Demand-Aware']
                ]
            }
        }

        results_file = os.path.join(output_dir, f'best_scenario_{num_sats}sats_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)

        print(f"结果数据: {results_file}")

    else:
        print("未找到有效场景")

    print("="*80)


if __name__ == "__main__":
    main()
