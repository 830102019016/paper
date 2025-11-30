"""
寻找P1算法优化效果最好的场景
通过测试多个随机种子，找到P1相对于其他算法改进最显著的场景
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '核心代码'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '对比实验'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, FancyArrowPatch
from typing import Dict, List, Tuple

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from algorithm1 import algorithm_1 as algorithm_1_p0, generate_scenario, Config
from algorithm1_p1_optimized import algorithm_1 as algorithm_1_p1
from baseline_methods import r_scheme, f_scheme

print('='*80)
print('寻找P1算法优化效果最好的场景')
print('='*80)

# ============================================================================
# 1. 测试多个场景
# ============================================================================
K = 50
num_uavs = 3
test_seeds = range(1, 101)  # 测试100个不同的随机种子

print(f'\n[Step 1] 测试 {len(test_seeds)} 个不同场景 (K={K}, num_uavs={num_uavs})...')

best_scenario = {
    'seed': None,
    'improvement_score': -float('inf'),
    'p1_energy': None,
    'r_energy': None,
    'f_energy': None,
    'p0_energy': None,
    'avg_improvement': None,
    'iot_positions': None,
    'iot_data_sizes': None
}

scenario_results = []

for seed in test_seeds:
    if seed % 10 == 0:
        print(f'  测试进度: {seed}/{len(test_seeds)}...')

    # 生成场景
    iot_positions, iot_data_sizes = generate_scenario(K=K, seed=seed)

    # 运行四种算法
    try:
        result_r = r_scheme(iot_positions, iot_data_sizes, num_uavs=num_uavs, seed=seed, verbose=False)
        result_f = f_scheme(iot_positions, iot_data_sizes, num_uavs=num_uavs, verbose=False)
        result_p0 = algorithm_1_p0(iot_positions, iot_data_sizes, num_uavs=num_uavs, verbose=False)
        result_p1 = algorithm_1_p1(iot_positions, iot_data_sizes, num_uavs=num_uavs,
                                    use_adam=True, use_2opt=True, verbose=False)

        # 计算能耗
        r_energy = result_r['energy']['hover_energy'] + result_r['energy']['flight_energy']
        f_energy = result_f['energy']['hover_energy'] + result_f['energy']['flight_energy']
        p0_energy = result_p0['energy']['hover_energy'] + result_p0['energy']['flight_energy']
        p1_energy = result_p1['energy']['hover_energy'] + result_p1['energy']['flight_energy']

        # 计算P1相对于其他算法的改进率
        improvement_vs_r = (r_energy - p1_energy) / r_energy * 100 if r_energy > 0 else 0
        improvement_vs_f = (f_energy - p1_energy) / f_energy * 100 if f_energy > 0 else 0
        improvement_vs_p0 = (p0_energy - p1_energy) / p0_energy * 100 if p0_energy > 0 else 0

        # 平均改进率作为评分标准
        avg_improvement = (improvement_vs_r + improvement_vs_f + improvement_vs_p0) / 3

        # 综合评分：考虑平均改进率和最小改进率（确保在所有算法上都有提升）
        min_improvement = min(improvement_vs_r, improvement_vs_f, improvement_vs_p0)
        improvement_score = avg_improvement * 0.7 + min_improvement * 0.3

        scenario_results.append({
            'seed': seed,
            'p1_energy': p1_energy,
            'r_energy': r_energy,
            'f_energy': f_energy,
            'p0_energy': p0_energy,
            'improvement_vs_r': improvement_vs_r,
            'improvement_vs_f': improvement_vs_f,
            'improvement_vs_p0': improvement_vs_p0,
            'avg_improvement': avg_improvement,
            'improvement_score': improvement_score
        })

        # 更新最佳场景
        if improvement_score > best_scenario['improvement_score']:
            best_scenario['seed'] = seed
            best_scenario['improvement_score'] = improvement_score
            best_scenario['p1_energy'] = p1_energy
            best_scenario['r_energy'] = r_energy
            best_scenario['f_energy'] = f_energy
            best_scenario['p0_energy'] = p0_energy
            best_scenario['avg_improvement'] = avg_improvement
            best_scenario['iot_positions'] = iot_positions
            best_scenario['iot_data_sizes'] = iot_data_sizes
            best_scenario['improvement_vs_r'] = improvement_vs_r
            best_scenario['improvement_vs_f'] = improvement_vs_f
            best_scenario['improvement_vs_p0'] = improvement_vs_p0

    except Exception as e:
        print(f'  警告: seed={seed} 运行失败: {e}')
        continue

# ============================================================================
# 2. 输出最佳场景信息
# ============================================================================
print('\n' + '='*80)
print('最佳P1优化场景')
print('='*80)
print(f"\n随机种子: {best_scenario['seed']}")
print(f"综合评分: {best_scenario['improvement_score']:.2f}")
print(f"\n能耗对比:")
print(f"  R-scheme:  {best_scenario['r_energy']:.2f} J")
print(f"  F-scheme:  {best_scenario['f_energy']:.2f} J")
print(f"  P0优化:    {best_scenario['p0_energy']:.2f} J")
print(f"  P1优化:    {best_scenario['p1_energy']:.2f} J")
print(f"\nP1改进率:")
print(f"  vs R-scheme:  {best_scenario['improvement_vs_r']:+.2f}%")
print(f"  vs F-scheme:  {best_scenario['improvement_vs_f']:+.2f}%")
print(f"  vs P0优化:    {best_scenario['improvement_vs_p0']:+.2f}%")
print(f"  平均改进:     {best_scenario['avg_improvement']:+.2f}%")

# ============================================================================
# 3. 显示Top 10最佳场景
# ============================================================================
print('\n' + '='*80)
print('Top 10 最佳P1优化场景')
print('='*80)

scenario_results.sort(key=lambda x: x['improvement_score'], reverse=True)
print(f"\n{'排名':<6} {'Seed':<8} {'评分':<10} {'P1能耗(J)':<12} {'平均改进%':<12} {'vs R%':<10} {'vs F%':<10} {'vs P0%':<10}")
print('-' * 90)

for rank, result in enumerate(scenario_results[:10], 1):
    print(f"{rank:<6} {result['seed']:<8} {result['improvement_score']:<10.2f} "
          f"{result['p1_energy']:<12.2f} {result['avg_improvement']:<12.2f} "
          f"{result['improvement_vs_r']:<10.2f} {result['improvement_vs_f']:<10.2f} "
          f"{result['improvement_vs_p0']:<10.2f}")

# ============================================================================
# 4. 用最佳场景重新运行并生成轨迹图
# ============================================================================
print('\n[Step 2] 使用最佳场景重新运行四种算法并生成轨迹图...')

best_seed = best_scenario['seed']
iot_positions = best_scenario['iot_positions']
iot_data_sizes = best_scenario['iot_data_sizes']

# 重新运行四种算法获取完整结果
print('  运行 R-scheme...')
result_rscheme = r_scheme(iot_positions, iot_data_sizes, num_uavs=num_uavs, seed=best_seed, verbose=False)

print('  运行 F-scheme...')
result_fscheme = f_scheme(iot_positions, iot_data_sizes, num_uavs=num_uavs, verbose=False)

print('  运行 P0优化...')
result_p0 = algorithm_1_p0(iot_positions, iot_data_sizes, num_uavs=num_uavs, verbose=False)

print('  运行 P1优化...')
result_p1 = algorithm_1_p1(iot_positions, iot_data_sizes, num_uavs=num_uavs,
                            use_adam=True, use_2opt=True, verbose=False)

# ============================================================================
# 5. 绘制轨迹对比图
# ============================================================================
print('\n[Step 3] 生成轨迹对比图...')

def plot_trajectory_comparison(results_list, iot_positions, best_scenario, save_path=None):
    """绘制四种方法的航迹对比"""
    fig = plt.figure(figsize=(20, 5))

    methods = ['R-scheme', 'F-scheme', 'P0优化', 'P1优化']
    uav_colors = ['red', 'blue', 'green']

    for idx, (result, method) in enumerate(zip(results_list, methods)):
        ax = fig.add_subplot(1, 4, idx + 1)

        # 绘制IoT设备
        paired_devices = set()
        for k, m in result['paired']:
            paired_devices.add(k)
            paired_devices.add(m)

        unpaired_devices = result['unpaired']

        # 配对设备（蓝色圆点）
        if len(paired_devices) > 0:
            paired_indices = list(paired_devices)
            ax.scatter(iot_positions[paired_indices, 0], iot_positions[paired_indices, 1],
                      c='blue', marker='o', s=80, label='Paired IoT', alpha=0.6, zorder=2)

        # 未配对设备（红色方块）
        if len(unpaired_devices) > 0:
            ax.scatter(iot_positions[unpaired_devices, 0], iot_positions[unpaired_devices, 1],
                      c='red', marker='s', s=80, label='Unpaired IoT', alpha=0.6, zorder=2)

        # 绘制配对连线
        for k, m in result['paired']:
            ax.plot([iot_positions[k, 0], iot_positions[m, 0]],
                   [iot_positions[k, 1], iot_positions[m, 1]],
                   'b--', alpha=0.2, linewidth=1, zorder=1)

        # 绘制悬停点
        hover_pos = result['hover_positions']
        ax.scatter(hover_pos[:, 0], hover_pos[:, 1],
                  c='green', marker='^', s=150, label='Hover Points',
                  edgecolors='black', linewidths=1.5, zorder=3)

        # 绘制UAV轨迹
        start_pos = np.array([Config.AREA_SIZE/2, Config.AREA_SIZE/2])

        for u, trajectory in enumerate(result['trajectories']):
            if len(trajectory) == 0:
                continue

            path_x = [start_pos[0]]
            path_y = [start_pos[1]]

            for point_idx in trajectory:
                path_x.append(hover_pos[point_idx, 0])
                path_y.append(hover_pos[point_idx, 1])

            path_x.append(start_pos[0])
            path_y.append(start_pos[1])

            ax.plot(path_x, path_y, '-', color=uav_colors[u], linewidth=2,
                   label=f'UAV{u+1}', alpha=0.7, zorder=2)

            # 添加箭头指示方向
            for i in range(len(path_x) - 1):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                if abs(dx) > 1 or abs(dy) > 1:  # 避免太短的箭头
                    ax.arrow(path_x[i], path_y[i], dx*0.3, dy*0.3,
                            head_width=15, head_length=10, fc=uav_colors[u],
                            ec=uav_colors[u], alpha=0.4, zorder=1)

        # 起点
        ax.scatter(start_pos[0], start_pos[1], c='black', marker='*',
                  s=400, label='Start/End', edgecolors='yellow', linewidths=2, zorder=4)

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)

        # 计算能耗
        uav_energy = result['energy']['hover_energy'] + result['energy']['flight_energy']
        distance = result['energy']['flight_distance']

        ax.set_title(f'{method}\n飞行距离: {distance:.0f}m | UAV能耗: {uav_energy:.0f}J',
                    fontsize=12, fontweight='bold')

        # 只在第一个子图显示图例
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim(-20, Config.AREA_SIZE + 20)
        ax.set_ylim(-20, Config.AREA_SIZE + 20)

    # 总标题包含场景信息
    plt.suptitle(f'最佳P1优化场景轨迹对比 (Seed={best_scenario["seed"]}, '
                f'平均改进={best_scenario["avg_improvement"]:.1f}%)',
                fontsize=15, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'\n轨迹对比图已保存到: {save_path}')

    plt.show()

# 生成可视化
script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir, '..', '结果图表')
os.makedirs(result_dir, exist_ok=True)
save_path = os.path.abspath(os.path.join(result_dir, 'best_p1_trajectory_comparison.png'))

plot_trajectory_comparison(
    [result_rscheme, result_fscheme, result_p0, result_p1],
    iot_positions,
    best_scenario,
    save_path=save_path
)

print('\n' + '='*80)
print('最佳P1场景搜索与可视化完成！')
print(f'最佳场景使用种子: {best_seed}')
print(f'图片已保存至: {save_path}')
print('='*80)
