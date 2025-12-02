"""
对比实验主程序 - 4算法对比
比较四种方法的性能：
1. R-scheme - 随机配对基线
2. F-scheme - 固定悬停基线
3. P0优化 - NOMA配对 + 功率优化 + 位置优化
4. P1优化 - P0 + Adam优化器 + 2-opt轨迹优化

评估指标：
- UAV能耗（优化目标）：悬停能耗 + 飞行能耗
- UAV飞行距离、悬停时间
- 算法运行时间
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '核心代码'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List
import time
import json

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
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # 数学公式使用STIX字体
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['grid.linewidth'] = 0.8
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markersize'] = 8

from algorithm1 import generate_scenario, Config
from algorithm1 import algorithm_1 as p0_algorithm_1
from baseline_methods import r_scheme, f_scheme
from algorithm1_p1_optimized import algorithm_1 as p1_algorithm_1


# ============================================================================
# 实验配置
# ============================================================================

class ExperimentConfig:
    """实验参数配置"""
    # 场景参数
    NUM_DEVICES_LIST = [20, 30, 40, 50]  # 不同设备数量
    NUM_UAVS = 3
    RANDOM_SEEDS = [27, 41, 37]  # Top 3 P1优化效果最好的场景

    # 固定场景（用于单次详细对比）
    FIXED_NUM_DEVICES = 30
    FIXED_SEED = 42


# ============================================================================
# 单次实验运行
# ============================================================================

def run_single_experiment_4algorithms(iot_positions: np.ndarray,
                                     iot_data_sizes: np.ndarray,
                                     num_uavs: int = 3,
                                     verbose: bool = False) -> Dict:
    """
    运行一次完整的4算法对比实验

    Args:
        iot_positions: IoT设备位置
        iot_data_sizes: 数据量
        num_uavs: UAV数量
        verbose: 是否打印详细信息

    Returns:
        results: {
            'r_scheme': {...},
            'f_scheme': {...},
            'p0': {...},
            'p1': {...}
        }
    """
    results = {}

    # 方法1: R-scheme
    if verbose:
        print("\n" + "="*70)
        print("运行方法 1/4: R-scheme (随机配对)")
        print("="*70)

    start_time = time.time()
    results['r_scheme'] = r_scheme(
        iot_positions, iot_data_sizes, num_uavs, seed=42, verbose=verbose
    )
    results['r_scheme']['runtime'] = time.time() - start_time
    results['r_scheme']['method'] = 'R-scheme'

    # 方法2: F-scheme
    if verbose:
        print("\n" + "="*70)
        print("运行方法 2/4: F-scheme (固定悬停)")
        print("="*70)

    start_time = time.time()
    results['f_scheme'] = f_scheme(
        iot_positions, iot_data_sizes, num_uavs, verbose=verbose
    )
    results['f_scheme']['runtime'] = time.time() - start_time
    results['f_scheme']['method'] = 'F-scheme'

    # 方法3: P0优化（基础版本）
    if verbose:
        print("\n" + "="*70)
        print("运行方法 3/4: P0优化 (基础优化)")
        print("="*70)

    start_time = time.time()
    results['p0'] = p0_algorithm_1(
        iot_positions, iot_data_sizes, num_uavs,
        use_gwo=False,
        verbose=verbose
    )
    results['p0']['runtime'] = time.time() - start_time
    results['p0']['method'] = 'P0优化'

    # 方法4: P1优化（带Adam和2-opt）
    if verbose:
        print("\n" + "="*70)
        print("运行方法 4/4: P1优化 (Adam + 2-opt)")
        print("="*70)

    start_time = time.time()
    results['p1'] = p1_algorithm_1(
        iot_positions, iot_data_sizes, num_uavs,
        use_gwo=False,
        use_adam=True,
        use_2opt=True,
        verbose=verbose
    )
    results['p1']['runtime'] = time.time() - start_time
    results['p1']['method'] = 'P1优化'

    return results


# ============================================================================
# 批量实验（变化设备数量）
# ============================================================================

def run_batch_experiments(verbose: bool = False) -> Dict:
    """
    批量实验：改变IoT设备数量

    Returns:
        batch_results: {
            'num_devices': [20, 30, 40, 50],
            'r_scheme': {...},
            'f_scheme': {...},
            'p0': {...},
            'p1': {...}
        }
    """
    batch_results = {
        'num_devices': ExperimentConfig.NUM_DEVICES_LIST,
        'r_scheme': {
            'uav_energy': [], 'hover_energy': [],
            'flight_energy': [], 'flight_distance': [], 'hover_time': [], 'runtime': []
        },
        'f_scheme': {
            'uav_energy': [], 'hover_energy': [],
            'flight_energy': [], 'flight_distance': [], 'hover_time': [], 'runtime': []
        },
        'p0': {
            'uav_energy': [], 'hover_energy': [],
            'flight_energy': [], 'flight_distance': [], 'hover_time': [], 'runtime': []
        },
        'p1': {
            'uav_energy': [], 'hover_energy': [],
            'flight_energy': [], 'flight_distance': [], 'hover_time': [], 'runtime': []
        }
    }

    print("\n" + "="*70)
    print("4算法批量对比实验")
    print("="*70)

    for K in ExperimentConfig.NUM_DEVICES_LIST:
        print(f"\n>>> 实验场景: K = {K} 设备")
        print("-" * 70)

        # 多次实验取平均
        avg_results = {
            'r_scheme': {'uav_energy': [], 'hover_energy': [],
                        'flight_energy': [], 'flight_distance': [], 'hover_time': [], 'runtime': []},
            'f_scheme': {'uav_energy': [], 'hover_energy': [],
                        'flight_energy': [], 'flight_distance': [], 'hover_time': [], 'runtime': []},
            'p0': {'uav_energy': [], 'hover_energy': [],
                   'flight_energy': [], 'flight_distance': [], 'hover_time': [], 'runtime': []},
            'p1': {'uav_energy': [], 'hover_energy': [],
                   'flight_energy': [], 'flight_distance': [], 'hover_time': [], 'runtime': []}
        }

        for seed in ExperimentConfig.RANDOM_SEEDS:
            # 生成场景
            iot_positions, iot_data_sizes = generate_scenario(K=K, seed=seed)

            # 运行实验
            results = run_single_experiment_4algorithms(
                iot_positions, iot_data_sizes,
                num_uavs=ExperimentConfig.NUM_UAVS,
                verbose=False
            )

            # 收集结果
            for method in ['r_scheme', 'f_scheme', 'p0', 'p1']:
                avg_results[method]['uav_energy'].append(results[method]['uav_energy'])
                avg_results[method]['hover_energy'].append(results[method]['energy']['hover_energy'])
                avg_results[method]['flight_energy'].append(results[method]['energy']['flight_energy'])
                avg_results[method]['flight_distance'].append(results[method]['energy']['flight_distance'])
                avg_results[method]['hover_time'].append(results[method]['energy']['hover_time'])
                avg_results[method]['runtime'].append(results[method]['runtime'])

        # 计算平均值
        for method in ['r_scheme', 'f_scheme', 'p0', 'p1']:
            batch_results[method]['uav_energy'].append(
                np.mean(avg_results[method]['uav_energy'])
            )
            batch_results[method]['hover_energy'].append(
                np.mean(avg_results[method]['hover_energy'])
            )
            batch_results[method]['flight_energy'].append(
                np.mean(avg_results[method]['flight_energy'])
            )
            batch_results[method]['flight_distance'].append(
                np.mean(avg_results[method]['flight_distance'])
            )
            batch_results[method]['hover_time'].append(
                np.mean(avg_results[method]['hover_time'])
            )
            batch_results[method]['runtime'].append(
                np.mean(avg_results[method]['runtime'])
            )

        # 打印当前结果（UAV能耗）
        print(f"  R-scheme:  {batch_results['r_scheme']['uav_energy'][-1]:.2f} J")
        print(f"  F-scheme:  {batch_results['f_scheme']['uav_energy'][-1]:.2f} J")
        print(f"  P0优化:    {batch_results['p0']['uav_energy'][-1]:.2f} J")
        print(f"  P1优化:    {batch_results['p1']['uav_energy'][-1]:.2f} J (Best)")

    print("\n" + "="*70)
    print("批量实验完成！")
    print("="*70)

    return batch_results


# ============================================================================
# 可视化函数
# ============================================================================

def plot_comparison_results(batch_results: Dict, save_path: str = None):
    """
    绘制4算法对比实验结果（2×2布局，4个子图）
    """
    fig = plt.figure(figsize=(14, 12))

    num_devices = batch_results['num_devices']
    methods = ['r_scheme', 'f_scheme', 'p0', 'p1']
    method_labels = {
        'r_scheme': 'Rand-NOMA',
        'f_scheme': 'Max-Hover',
        'p0': 'Joint-Opt',
        'p1': 'Adam-2opt'
    }
    colors = {
        'r_scheme': '#9b59b6',  # 紫色
        'f_scheme': '#95a5a6',  # 灰色
        'p0': '#3498db',        # 蓝色
        'p1': '#e74c3c'         # 红色
    }
    markers = {
        'r_scheme': 's',
        'f_scheme': '^',
        'p0': 'o',
        'p1': '*'  # 星形
    }

    # 子图1: UAV总能耗（优化目标）- 左上
    ax1 = fig.add_subplot(2, 2, 1)
    for method in methods:
        ax1.plot(num_devices, batch_results[method]['uav_energy'],
                marker=markers[method], linewidth=2.5, markersize=11,
                label=method_labels[method], color=colors[method], alpha=0.9)
    ax1.set_xlabel('Number of IoT Devices $K$\n(a) Total UAV Energy', fontsize=13)
    ax1.set_ylabel('UAV Energy (J)', fontsize=13)
    ax1.legend(fontsize=10.5, framealpha=0.95, edgecolor='black',
               fancybox=False, shadow=False, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.tick_params(direction='in', which='both')

    # 子图2: 飞行能耗 - 右上
    ax2 = fig.add_subplot(2, 2, 2)
    for method in methods:
        ax2.plot(num_devices, batch_results[method]['flight_energy'],
                marker=markers[method], linewidth=2.5, markersize=11,
                label=method_labels[method], color=colors[method], alpha=0.9)
    ax2.set_xlabel('Number of IoT Devices $K$\n(b) Flight Energy', fontsize=13)
    ax2.set_ylabel('Flight Energy (J)', fontsize=13)
    ax2.legend(fontsize=10.5, framealpha=0.95, edgecolor='black',
               fancybox=False, shadow=False, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.tick_params(direction='in', which='both')

    # 子图3: 飞行距离 - 左下
    ax3 = fig.add_subplot(2, 2, 3)
    for method in methods:
        ax3.plot(num_devices, batch_results[method]['flight_distance'],
                marker=markers[method], linewidth=2.5, markersize=11,
                label=method_labels[method], color=colors[method], alpha=0.9)
    ax3.set_xlabel('Number of IoT Devices $K$\n(c) Flight Distance', fontsize=13)
    ax3.set_ylabel('Flight Distance (m)', fontsize=13)
    ax3.legend(fontsize=10.5, framealpha=0.95, edgecolor='black',
               fancybox=False, shadow=False, loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.tick_params(direction='in', which='both')

    # 子图4: 悬停能耗 - 右下
    ax4 = fig.add_subplot(2, 2, 4)
    for method in methods:
        ax4.plot(num_devices, batch_results[method]['hover_energy'],
                marker=markers[method], linewidth=2.5, markersize=11,
                label=method_labels[method], color=colors[method], alpha=0.9)
    ax4.set_xlabel('Number of IoT Devices $K$\n(d) Hovering Energy', fontsize=13)
    ax4.set_ylabel('Hovering Energy (J)', fontsize=13)
    ax4.legend(fontsize=10.5, framealpha=0.95, edgecolor='black',
               fancybox=False, shadow=False, loc='best')
    ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax4.tick_params(direction='in', which='both')

    plt.tight_layout()

    if save_path:
        # 保存PNG格式
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存至: {save_path}")

        # 保存EPS格式（矢量图，适合论文发表）
        eps_path = save_path.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight')
        print(f"EPS格式已保存至: {eps_path}")

    plt.show()


def print_performance_summary(batch_results: Dict):
    """
    打印性能总结表格
    """
    print("\n" + "="*100)
    print("4算法性能总结 (Performance Summary)")
    print("="*100)

    for i, K in enumerate(batch_results['num_devices']):
        print(f"\n设备数量 K = {K}")
        print("-" * 80)
        print(f"{'方法':<15} {'UAV能耗(J)':<12} {'飞行距离(m)':<15} {'悬停时间(s)':<15} {'时间(s)':<10}")
        print("-" * 80)

        for method, label in [('r_scheme', 'R-scheme'),
                             ('f_scheme', 'F-scheme'),
                             ('p0', 'P0 Opt'),
                             ('p1', 'P1 Opt (Best)')]:
            uav_energy = batch_results[method]['uav_energy'][i]
            dist = batch_results[method]['flight_distance'][i]
            hover_t = batch_results[method]['hover_time'][i]
            time_val = batch_results[method]['runtime'][i]

            print(f"{label:<15} {uav_energy:<12.2f} {dist:<15.2f} {hover_t:<15.2f} {time_val:<10.3f}")

        # 计算改进（基于UAV能耗）
        r_uav = batch_results['r_scheme']['uav_energy'][i]
        f_uav = batch_results['f_scheme']['uav_energy'][i]
        p0_uav = batch_results['p0']['uav_energy'][i]
        p1_uav = batch_results['p1']['uav_energy'][i]

        p1_vs_r = (r_uav - p1_uav) / r_uav * 100
        p1_vs_f = (f_uav - p1_uav) / f_uav * 100
        p1_vs_p0 = (p0_uav - p1_uav) / p0_uav * 100

        print("-" * 80)
        print(f"P1改进: vs R-scheme: {p1_vs_r:+.2f}%, vs F-scheme: {p1_vs_f:+.2f}%, vs P0: {p1_vs_p0:+.2f}%")

    print("\n" + "="*100)


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    print("\n" + "="*80)
    print("IoT-UAV 数据收集系统 - 4算法对比实验")
    print("比较方法: R-scheme, F-scheme, P0优化, P1优化")
    print("="*80)

    # 运行批量实验
    batch_results = run_batch_experiments(verbose=False)

    # 打印性能总结
    print_performance_summary(batch_results)

    # 绘制对比图表
    print("\n生成对比图表...")
    save_dir = os.path.join(os.path.dirname(__file__), '..', '结果图表')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'comparison_4algorithms.png')
    plot_comparison_results(batch_results, save_path=save_path)

    # 保存结果到JSON
    print("\n保存实验结果...")
    json_path = os.path.join(save_dir, 'experiment_4algorithms_results.json')
    with open(json_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    print(f"结果已保存至: {json_path}")

    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)

    return batch_results


if __name__ == "__main__":
    results = main()
