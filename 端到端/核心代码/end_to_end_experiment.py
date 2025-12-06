"""
端到端实验主程序 (End-to-End Experiment)
联合评估 Algorithm 1 (IoT-UAV) + Algorithm 2 (UAV-LEO)

实验设计：
- 4个对照组：G1(P1+DA), G2(P1+Greedy), G3(F-scheme+DA), G4(F-scheme+Greedy)
- 3个核心指标：端到端总时延、系统总能耗、数据传输成功率

作者: 实验代码
日期: 2025-12-06
"""

import numpy as np
import sys
import os
from typing import Dict, Tuple, List
from datetime import datetime, timezone, timedelta
import time

# 添加算法1和算法2的路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../算法1/核心代码'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../算法2/核心代码'))

# 导入算法1的模块
try:
    from algorithm1_p1_optimized import algorithm_1 as algorithm_1_p1
    from baseline_methods import f_scheme
    import algorithm1_p1_optimized as alg1_module
except ImportError as e:
    print(f"[错误] 无法导入算法1模块: {e}")
    sys.exit(1)

# 导入算法2的模块
try:
    from algorithm2_demand_aware import (
        LEOConstellation,
        UAVLocation,
        GreedySelector,
        DemandAwareSelector,
        TransmissionRecord
    )
except ImportError as e:
    print(f"[错误] 无法导入算法2模块: {e}")
    sys.exit(1)

# ============================================================================
# 全局配置
# ============================================================================

class E2EConfig:
    """端到端实验配置"""

    # 算法1参数 (IoT-UAV阶段)
    NUM_UAVS = 3
    UAV_ALTITUDE_M = 200.0  # 与算法2一致
    AREA_SIZE = 500.0  # m

    # 算法2参数 (UAV-LEO阶段)
    UAV_LAT = 15.0  # 度
    UAV_LON = 118.0  # 度
    NUM_SATELLITES = 1500  # 卫星数量
    DEMAND_RATE_MBPS = 8.0  # 需求速率 (1080p视频流)
    TRANSMISSION_DURATION_MIN = 10  # 传输时长(分钟)
    TIME_STEP_SEC = 20  # 采样间隔(秒)

    # 能耗参数
    IOT_TRANSMISSION_POWER_W = 0.1  # IoT设备传输功率(W) - 简化为定值
    SATELLITE_COMPUTING_POWER_W = 50.0  # 卫星计算功率(W) - 典型LEO卫星计算功耗

    # 切换开销
    HANDOVER_DELAY_SEC = 0.4  # 切换时延(秒)

    # 实验参数
    RANDOM_SEEDS = [27, 37, 41]  # 3个随机种子
    K_VALUES = [40]  # IoT设备数量 (主实验固定K=40)

    # TLE文件路径
    TLE_FILE = 'starlink.tle'


# ============================================================================
# 阶段1：算法1 (IoT-UAV数据收集)
# ============================================================================

def run_algorithm1_phase(iot_positions: np.ndarray,
                        iot_data_sizes: np.ndarray,
                        method: str = 'P1') -> Dict:
    """
    运行算法1阶段

    Args:
        iot_positions: IoT位置 [K, 3]
        iot_data_sizes: 数据量 [K] (bits)
        method: 'P1' 或 'F-scheme'

    Returns:
        result: {
            'method': 方法名称,
            'collection_time': 收集时间(秒),
            'uav_energy': UAV能耗(焦耳),
            'iot_energy': IoT能耗(焦耳),
            'data_collected': 收集的数据量(MB),
            'hover_time': 悬停时间(秒),
            'flight_time': 飞行时间(秒),
            'algorithm1_result': 算法1原始输出
        }
    """
    print(f"  [阶段1] 运行算法1 ({method})...")

    if method == 'P1':
        # 使用P1优化算法
        alg1_result = algorithm_1_p1(
            iot_positions=iot_positions,
            iot_data_sizes=iot_data_sizes,
            num_uavs=E2EConfig.NUM_UAVS,
            use_gwo=False,  # 使用2-opt优化
            verbose=False
        )
    elif method == 'F-scheme':
        # 使用F-scheme基线
        alg1_result = f_scheme(
            iot_positions=iot_positions,
            iot_data_sizes=iot_data_sizes,
            num_uavs=E2EConfig.NUM_UAVS,
            verbose=False
        )
    else:
        raise ValueError(f"未知的算法1方法: {method}")

    # 提取关键信息
    energy_dict = alg1_result['energy']
    hover_time = energy_dict['hover_time']  # 秒
    flight_distance = energy_dict['flight_distance']  # 米

    # 计算飞行时间
    flight_speed = 20.0  # m/s (算法1设定)
    flight_time = flight_distance / flight_speed  # 秒

    # 总收集时间
    collection_time = hover_time + flight_time

    # UAV能耗
    uav_energy = alg1_result['uav_energy']  # 焦耳

    # 计算IoT传输能耗 (简化计算)
    # E_IoT = P_IoT × T_hover
    iot_energy = E2EConfig.IOT_TRANSMISSION_POWER_W * hover_time  # 焦耳

    # 收集的数据量 (转换为MB)
    data_collected_bits = np.sum(iot_data_sizes)
    data_collected_mb = data_collected_bits / 8 / (1024 ** 2)

    result = {
        'method': method,
        'collection_time': collection_time,
        'uav_energy': uav_energy,
        'iot_energy': iot_energy,
        'data_collected': data_collected_mb,
        'hover_time': hover_time,
        'flight_time': flight_time,
        'algorithm1_result': alg1_result
    }

    print(f"    收集时间: {collection_time:.2f}s, UAV能耗: {uav_energy:.2f}J, 数据量: {data_collected_mb:.2f}MB")

    return result


# ============================================================================
# 阶段2：算法2 (UAV-LEO卫星传输)
# ============================================================================

def run_algorithm2_phase(data_to_transmit_mb: float,
                        method: str = 'Demand-Aware',
                        start_time: datetime = None) -> Dict:
    """
    运行算法2阶段

    Args:
        data_to_transmit_mb: 需要传输的数据量(MB)
        method: 'Demand-Aware' 或 'Greedy'
        start_time: 起始时间

    Returns:
        result: {
            'method': 方法名称,
            'transmission_time': 有效传输时间(秒),
            'handover_count': 切换次数,
            'handover_overhead': 切换开销时间(秒),
            'packet_loss_rate': 丢包率,
            'avg_data_rate': 平均数据速率(Mbps),
            'data_transmitted': 实际传输的数据量(MB),
            'satellite_energy': 卫星计算能耗(焦耳),
            'records': 传输记录列表
        }
    """
    print(f"  [阶段2] 运行算法2 ({method})...")

    # 检查TLE文件
    if not os.path.exists(E2EConfig.TLE_FILE):
        raise FileNotFoundError(f"TLE文件未找到: {E2EConfig.TLE_FILE}\n请先下载starlink.tle文件")

    # 创建LEO星座
    constellation = LEOConstellation(E2EConfig.TLE_FILE, E2EConfig.NUM_SATELLITES)

    # 创建选择器
    if method == 'Demand-Aware':
        selector = DemandAwareSelector(
            constellation,
            demand_mbps=E2EConfig.DEMAND_RATE_MBPS
        )
    elif method == 'Greedy':
        selector = GreedySelector(constellation)
    else:
        raise ValueError(f"未知的算法2方法: {method}")

    # UAV位置
    uav_loc = UAVLocation(
        lat=E2EConfig.UAV_LAT,
        lon=E2EConfig.UAV_LON,
        alt_m=E2EConfig.UAV_ALTITUDE_M
    )

    # 使用固定起始时间
    if start_time is None:
        start_time = datetime.now(timezone.utc)

    # 仿真时长
    duration_sec = E2EConfig.TRANSMISSION_DURATION_MIN * 60
    num_steps = duration_sec // E2EConfig.TIME_STEP_SEC

    # 记录
    records = []
    cumulative_data_mb = 0.0

    for step in range(num_steps):
        current_time = start_time + timedelta(seconds=step * E2EConfig.TIME_STEP_SEC)
        time_utc = current_time.strftime('%Y-%m-%d %H:%M:%S')
        elapsed_sec = step * E2EConfig.TIME_STEP_SEC

        # 选择卫星
        sat = selector.select(uav_loc, time_utc)

        if sat is not None:
            data_rate_mbps = sat.data_rate_mbps
            # 传输的数据量 (Mbps × 时间步长(秒) / 8 = MB)
            transmitted_mb = data_rate_mbps * E2EConfig.TIME_STEP_SEC / 8
            cumulative_data_mb += transmitted_mb

            record = TransmissionRecord(
                time_sec=elapsed_sec,
                data_rate_mbps=data_rate_mbps,
                cumulative_data_mb=cumulative_data_mb,
                satellite_id=sat.sat_id,
                distance_km=sat.distance_km
            )
        else:
            # 无可用卫星
            record = TransmissionRecord(
                time_sec=elapsed_sec,
                data_rate_mbps=0.0,
                cumulative_data_mb=cumulative_data_mb,
                satellite_id=-1,
                distance_km=0.0
            )

        records.append(record)

    # 统计切换次数
    handover_count = 0
    prev_sat_id = None
    for r in records:
        if prev_sat_id is not None and r.satellite_id != prev_sat_id and r.satellite_id != -1:
            handover_count += 1
        prev_sat_id = r.satellite_id

    # 切换开销时间
    handover_overhead = handover_count * E2EConfig.HANDOVER_DELAY_SEC

    # 有效传输时间
    transmission_time = duration_sec - handover_overhead

    # 丢包率 (切换时间占比)
    packet_loss_rate = handover_overhead / duration_sec

    # 平均数据速率
    rates = [r.data_rate_mbps for r in records if r.data_rate_mbps > 0]
    avg_data_rate = np.mean(rates) if len(rates) > 0 else 0.0

    # 实际传输的数据量 (考虑丢包)
    data_transmitted = cumulative_data_mb * (1 - packet_loss_rate)

    # 卫星计算能耗
    # E_satellite = P_computing × T_transmission
    # 卫星在整个传输期间需要持续计算处理数据
    satellite_energy = E2EConfig.SATELLITE_COMPUTING_POWER_W * duration_sec  # 焦耳

    result = {
        'method': method,
        'transmission_time': transmission_time,
        'handover_count': handover_count,
        'handover_overhead': handover_overhead,
        'packet_loss_rate': packet_loss_rate,
        'avg_data_rate': avg_data_rate,
        'data_transmitted': data_transmitted,
        'satellite_energy': satellite_energy,
        'records': records
    }

    print(f"    切换次数: {handover_count}, 丢包率: {packet_loss_rate*100:.3f}%, "
          f"平均速率: {avg_data_rate:.2f}Mbps, 传输量: {data_transmitted:.2f}MB, "
          f"卫星能耗: {satellite_energy:.2f}J")

    return result


# ============================================================================
# 端到端指标计算
# ============================================================================

def calculate_e2e_metrics(phase1_result: Dict, phase2_result: Dict) -> Dict:
    """
    计算端到端性能指标

    Args:
        phase1_result: 阶段1结果
        phase2_result: 阶段2结果

    Returns:
        metrics: {
            'e2e_latency': 端到端总时延(秒),
            'system_energy': 系统总能耗(焦耳),
            'delivery_success_rate': 数据传输成功率
        }
    """
    # 1. 端到端总时延
    # T_E2E = T_collection + T_transmission + T_handover_overhead
    e2e_latency = (phase1_result['collection_time'] +
                   phase2_result['transmission_time'] +
                   phase2_result['handover_overhead'])

    # 2. 系统总能耗
    # E_system = E_UAV + E_IoT + E_satellite
    system_energy = (phase1_result['uav_energy'] +
                    phase1_result['iot_energy'] +
                    phase2_result['satellite_energy'])

    # 3. 数据传输成功率
    # η_delivery = (D_collected × (1 - PLR)) / D_collected = (1 - PLR)
    # 简化：假设算法1成功收集所有数据，成功率主要取决于算法2的丢包率
    delivery_success_rate = 1 - phase2_result['packet_loss_rate']

    # 也可以考虑传输的数据量与收集的数据量的比值
    # 但在10分钟内，通常能传输完所有数据（8Mbps × 600s / 8 = 600MB >> 实际数据量）

    metrics = {
        'e2e_latency': e2e_latency,
        'system_energy': system_energy,
        'delivery_success_rate': delivery_success_rate
    }

    return metrics


# ============================================================================
# 单次实验运行
# ============================================================================

def run_single_experiment(group_id: str,
                         alg1_method: str,
                         alg2_method: str,
                         K: int,
                         seed: int,
                         start_time: datetime = None) -> Dict:
    """
    运行单次端到端实验

    Args:
        group_id: 组别ID (G1, G2, G3, G4)
        alg1_method: 算法1方法 ('P1' 或 'F-scheme')
        alg2_method: 算法2方法 ('Demand-Aware' 或 'Greedy')
        K: IoT设备数量
        seed: 随机种子
        start_time: 算法2的起始时间

    Returns:
        result: 完整实验结果
    """
    print(f"\n{'='*70}")
    print(f"运行实验: {group_id} | 算法1={alg1_method}, 算法2={alg2_method} | K={K}, Seed={seed}")
    print(f"{'='*70}")

    # 生成IoT场景
    np.random.seed(seed)
    iot_positions = np.random.uniform(0, E2EConfig.AREA_SIZE, (K, 2))
    iot_positions = np.column_stack([iot_positions, np.zeros(K)])

    # 数据包大小: 100KB - 1MB (转换为bits)
    data_size_min = 100e3 * 8
    data_size_max = 1e6 * 8
    iot_data_sizes = np.random.uniform(data_size_min, data_size_max, K)

    # 阶段1: 算法1 (IoT-UAV数据收集)
    phase1_result = run_algorithm1_phase(iot_positions, iot_data_sizes, alg1_method)

    # 阶段2: 算法2 (UAV-LEO卫星传输)
    phase2_result = run_algorithm2_phase(
        phase1_result['data_collected'],
        alg2_method,
        start_time
    )

    # 计算端到端指标
    e2e_metrics = calculate_e2e_metrics(phase1_result, phase2_result)

    # 汇总结果
    result = {
        'group_id': group_id,
        'alg1_method': alg1_method,
        'alg2_method': alg2_method,
        'K': K,
        'seed': seed,
        'phase1': phase1_result,
        'phase2': phase2_result,
        'e2e_metrics': e2e_metrics
    }

    # 打印端到端指标
    print(f"\n[端到端指标]")
    print(f"  总时延: {e2e_metrics['e2e_latency']:.2f}s")
    print(f"  总能耗: {e2e_metrics['system_energy']:.2f}J")
    print(f"  成功率: {e2e_metrics['delivery_success_rate']*100:.3f}%")

    return result


# ============================================================================
# 批量实验运行
# ============================================================================

def run_all_experiments(K_values: List[int] = None,
                       seeds: List[int] = None) -> Dict:
    """
    运行所有对照组实验

    Args:
        K_values: IoT设备数量列表
        seeds: 随机种子列表

    Returns:
        all_results: {
            'G1': [实验结果列表],
            'G2': [...],
            'G3': [...],
            'G4': [...]
        }
    """
    if K_values is None:
        K_values = E2EConfig.K_VALUES
    if seeds is None:
        seeds = E2EConfig.RANDOM_SEEDS

    # 定义4个对照组
    groups = {
        'G1': {'alg1': 'P1', 'alg2': 'Demand-Aware', 'desc': 'P1 + Demand-Aware (提出方法)'},
        'G2': {'alg1': 'P1', 'alg2': 'Greedy', 'desc': 'P1 + Greedy (半优化A)'},
        'G3': {'alg1': 'F-scheme', 'alg2': 'Demand-Aware', 'desc': 'F-scheme + Demand-Aware (半优化B)'},
        'G4': {'alg1': 'F-scheme', 'alg2': 'Greedy', 'desc': 'F-scheme + Greedy (传统基线)'}
    }

    # 存储所有结果
    all_results = {gid: [] for gid in groups.keys()}

    # 使用固定的起始时间确保算法2的可比性
    start_time = datetime.now(timezone.utc)

    print("\n" + "="*80)
    print("端到端实验 - 4组对照实验")
    print("="*80)
    print(f"IoT设备数量: {K_values}")
    print(f"随机种子: {seeds}")
    print(f"算法2起始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("\n对照组配置:")
    for gid, config in groups.items():
        print(f"  {gid}: {config['desc']}")
    print("="*80)

    # 运行实验
    total_experiments = len(groups) * len(K_values) * len(seeds)
    experiment_count = 0

    for K in K_values:
        for seed in seeds:
            for gid, config in groups.items():
                experiment_count += 1
                print(f"\n[进度: {experiment_count}/{total_experiments}]")

                try:
                    result = run_single_experiment(
                        group_id=gid,
                        alg1_method=config['alg1'],
                        alg2_method=config['alg2'],
                        K=K,
                        seed=seed,
                        start_time=start_time
                    )
                    all_results[gid].append(result)

                except Exception as e:
                    print(f"[错误] 实验失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print("\n" + "="*80)
    print("所有实验完成!")
    print("="*80)

    return all_results


# ============================================================================
# 结果统计与分析
# ============================================================================

def analyze_results(all_results: Dict) -> Dict:
    """
    统计分析实验结果

    Args:
        all_results: 所有实验结果

    Returns:
        statistics: 统计结果
    """
    print("\n" + "="*80)
    print("实验结果统计分析")
    print("="*80)

    statistics = {}

    for gid in ['G1', 'G2', 'G3', 'G4']:
        results = all_results[gid]

        if len(results) == 0:
            continue

        # 提取指标
        e2e_latencies = [r['e2e_metrics']['e2e_latency'] for r in results]
        system_energies = [r['e2e_metrics']['system_energy'] for r in results]
        delivery_rates = [r['e2e_metrics']['delivery_success_rate'] for r in results]

        # 计算统计量
        stats = {
            'group_id': gid,
            'n_experiments': len(results),
            'e2e_latency_mean': np.mean(e2e_latencies),
            'e2e_latency_std': np.std(e2e_latencies),
            'system_energy_mean': np.mean(system_energies),
            'system_energy_std': np.std(system_energies),
            'delivery_rate_mean': np.mean(delivery_rates),
            'delivery_rate_std': np.std(delivery_rates)
        }

        statistics[gid] = stats

        # 打印结果
        print(f"\n{gid} 统计 (n={stats['n_experiments']}):")
        print(f"  端到端时延: {stats['e2e_latency_mean']:.2f} ± {stats['e2e_latency_std']:.2f} s")
        print(f"  系统能耗:   {stats['system_energy_mean']:.2f} ± {stats['system_energy_std']:.2f} J")
        print(f"  成功率:     {stats['delivery_rate_mean']*100:.3f} ± {stats['delivery_rate_std']*100:.3f} %")

    # 对比分析 (G1 vs 其他组)
    if 'G1' in statistics:
        print("\n" + "-"*80)
        print("G1 (提出方法) 相对改进:")
        print("-"*80)

        g1_stats = statistics['G1']

        for gid in ['G2', 'G3', 'G4']:
            if gid not in statistics:
                continue

            g_stats = statistics[gid]

            # 计算改进率 (负值表示G1更好)
            latency_improvement = (g1_stats['e2e_latency_mean'] - g_stats['e2e_latency_mean']) / g_stats['e2e_latency_mean'] * 100
            energy_improvement = (g1_stats['system_energy_mean'] - g_stats['system_energy_mean']) / g_stats['system_energy_mean'] * 100
            rate_improvement = (g1_stats['delivery_rate_mean'] - g_stats['delivery_rate_mean']) * 100  # 绝对值

            print(f"\nG1 vs {gid}:")
            print(f"  时延: {latency_improvement:+.2f}% {'✓' if latency_improvement < 0 else '✗'}")
            print(f"  能耗: {energy_improvement:+.2f}% {'✓' if energy_improvement < 0 else '✗'}")
            print(f"  成功率: {rate_improvement:+.3f}% (绝对值) {'✓' if rate_improvement > 0 else '✗'}")

    print("\n" + "="*80)

    return statistics


# ============================================================================
# 保存结果
# ============================================================================

def save_results(all_results: Dict, statistics: Dict, output_dir: str = None):
    """
    保存实验结果到JSON文件

    Args:
        all_results: 所有实验结果
        statistics: 统计结果
        output_dir: 输出目录
    """
    import json

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '../结果图表')

    os.makedirs(output_dir, exist_ok=True)

    # 准备可序列化的数据
    save_data = {
        'metadata': {
            'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'k_values': E2EConfig.K_VALUES,
            'random_seeds': E2EConfig.RANDOM_SEEDS,
            'num_satellites': E2EConfig.NUM_SATELLITES,
            'demand_rate_mbps': E2EConfig.DEMAND_RATE_MBPS
        },
        'statistics': statistics,
        'raw_results': {}
    }

    # 简化原始结果 (移除不可序列化的对象)
    for gid, results in all_results.items():
        save_data['raw_results'][gid] = []
        for r in results:
            simplified = {
                'group_id': r['group_id'],
                'K': r['K'],
                'seed': r['seed'],
                'e2e_metrics': r['e2e_metrics'],
                'phase1_summary': {
                    'method': r['phase1']['method'],
                    'collection_time': r['phase1']['collection_time'],
                    'uav_energy': r['phase1']['uav_energy'],
                    'data_collected': r['phase1']['data_collected']
                },
                'phase2_summary': {
                    'method': r['phase2']['method'],
                    'handover_count': r['phase2']['handover_count'],
                    'packet_loss_rate': r['phase2']['packet_loss_rate'],
                    'avg_data_rate': r['phase2']['avg_data_rate'],
                    'satellite_energy': r['phase2']['satellite_energy']
                }
            }
            save_data['raw_results'][gid].append(simplified)

    # 保存为JSON
    output_file = os.path.join(output_dir, 'e2e_experiment_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n[保存] 结果已保存到: {output_file}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("\n")
    print("+" + "="*78 + "+")
    print("|" + " "*20 + "端到端实验主程序" + " "*40 + "|")
    print("|" + " "*10 + "Algorithm 1 (IoT-UAV) + Algorithm 2 (UAV-LEO)" + " "*21 + "|")
    print("+" + "="*78 + "+")

    start_time_total = time.time()

    try:
        # 运行所有实验
        all_results = run_all_experiments()

        # 统计分析
        statistics = analyze_results(all_results)

        # 保存结果
        save_results(all_results, statistics)

        elapsed = time.time() - start_time_total
        print(f"\n总耗时: {elapsed:.2f}s")
        print("\n实验完成! ✓")

        return all_results, statistics

    except Exception as e:
        print(f"\n[错误] 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    all_results, statistics = main()
