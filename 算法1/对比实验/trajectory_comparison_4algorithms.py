"""
航迹诊断与可视化对比：四种算法对比
详细对比R-scheme, F-scheme, P0优化, P1优化的航迹规划效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '核心代码'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '对比实验'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, FancyArrowPatch
from typing import Dict, List

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

from algorithm1 import algorithm_1 as algorithm_1_p0, generate_scenario, Config
from algorithm1_p1_optimized import algorithm_1 as algorithm_1_p1
from baseline_methods import r_scheme, f_scheme

print('='*80)
print('航迹诊断与可视化对比：R-scheme vs F-scheme vs P0 vs P1')
print('='*80)

# ============================================================================
# 1. 生成测试场景
# ============================================================================
K = 50
seed = 27  # 使用P1优化效果最好的场景
print(f'\n[Step 1] 生成测试场景: K={K}, seed={seed}')
iot_positions, iot_data_sizes = generate_scenario(K=K, seed=seed)

# ============================================================================
# 2. 运行四种方法
# ============================================================================
print('\n[Step 2] 运行四种方法...')

# R-scheme
print('  [1/4] R-scheme（随机配对基线）...')
result_rscheme = r_scheme(iot_positions, iot_data_sizes, num_uavs=3, seed=seed, verbose=False)

# F-scheme
print('  [2/4] F-scheme（固定悬停基线）...')
result_fscheme = f_scheme(iot_positions, iot_data_sizes, num_uavs=3, verbose=False)

# P0优化版
print('  [3/4] P0优化版（基础优化）...')
result_p0 = algorithm_1_p0(iot_positions, iot_data_sizes, num_uavs=3, verbose=False)

# P1优化版（完整优化）
print('  [4/4] P1优化版（Adam + 2-opt）...')
result_p1 = algorithm_1_p1(
    iot_positions, iot_data_sizes,
    num_uavs=3,
    use_adam=True,
    use_2opt=True,
    verbose=False
)

# ============================================================================
# 3. 计算航迹统计
# ============================================================================
print('\n[Step 3] 计算航迹统计...')

def compute_trajectory_stats(result: Dict, method_name: str) -> Dict:
    """计算航迹详细统计"""
    stats = {
        'method': method_name,
        'hover_energy': result['energy']['hover_energy'],
        'flight_energy': result['energy']['flight_energy'],
        'flight_distance': result['energy']['flight_distance'],
        'hover_time': result['energy']['hover_time'],
        'num_paired': len(result['paired']),
        'num_unpaired': len(result['unpaired']),
        'pairing_rate': len(result['paired']) * 2 / K * 100,
        'per_uav': []
    }

    # 每个UAV的统计
    if 'per_uav_stats' in result['energy']:
        for uav_stat in result['energy']['per_uav_stats']:
            stats['per_uav'].append({
                'uav_id': uav_stat['uav_id'],
                'num_hover': uav_stat['num_hover_points'],
                'num_devices': uav_stat['num_iot_devices'],
                'distance': uav_stat['flight_distance'],
                'energy': uav_stat['flight_energy'],
                'efficiency': uav_stat['service_efficiency'],
                'trajectory': uav_stat['trajectory']
            })
    else:
        # 如果没有per_uav_stats，从trajectories中提取
        num_uavs = len(result['trajectories'])

        for u in range(num_uavs):
            trajectory = result['trajectories'][u]
            num_hover = len(trajectory)

            # 计算该UAV访问的设备数
            num_devices = 0
            if 'iot_assignments' in result:
                for hover_idx in trajectory:
                    if hover_idx < len(result['iot_assignments']):
                        num_devices += len(result['iot_assignments'][hover_idx])

            stats['per_uav'].append({
                'uav_id': u + 1,
                'num_hover': num_hover,
                'num_devices': num_devices if num_devices > 0 else 0,
                'distance': 0,  # 没有详细数据时设为0
                'energy': 0,
                'efficiency': num_devices / num_hover if num_hover > 0 else 0,
                'trajectory': trajectory
            })

    return stats

stats_rscheme = compute_trajectory_stats(result_rscheme, 'R-scheme')
stats_fscheme = compute_trajectory_stats(result_fscheme, 'F-scheme')
stats_p0 = compute_trajectory_stats(result_p0, 'P0优化')
stats_p1 = compute_trajectory_stats(result_p1, 'P1优化')

# ============================================================================
# 4. 打印对比统计
# ============================================================================
print('\n' + '='*100)
print('航迹对比统计')
print('='*100)

print(f"\n{'指标':<25} {'R-scheme':<15} {'F-scheme':<15} {'P0优化':<15} {'P1优化':<15}")
print('-' * 100)

metrics = [
    ('UAV总能耗 (J)', 'uav_energy'),
    ('飞行能耗 (J)', 'flight_energy'),
    ('飞行距离 (m)', 'flight_distance'),
    ('悬停时间 (s)', 'hover_time')
]

for label, key in metrics:
    # 处理uav_energy特殊情况
    if key == 'uav_energy':
        r_val = stats_rscheme['hover_energy'] + stats_rscheme['flight_energy']
        f_val = stats_fscheme['hover_energy'] + stats_fscheme['flight_energy']
        p0_val = stats_p0['hover_energy'] + stats_p0['flight_energy']
        p1_val = stats_p1['hover_energy'] + stats_p1['flight_energy']
    else:
        r_val = stats_rscheme[key]
        f_val = stats_fscheme[key]
        p0_val = stats_p0[key]
        p1_val = stats_p1[key]
    print(f"{label:<25} {r_val:<15.2f} {f_val:<15.2f} {p0_val:<15.2f} {p1_val:<15.2f}")

# 每个UAV的对比
print('\n' + '='*100)
print('每个UAV的航迹统计')
print('='*100)

for i in range(3):
    print(f'\n--- UAV {i+1} ---')
    print(f"{'指标':<20} {'R-scheme':<15} {'F-scheme':<15} {'P0优化':<15} {'P1优化':<15}")
    print('-' * 100)

    r_uav = stats_rscheme['per_uav'][i]
    f_uav = stats_fscheme['per_uav'][i]
    p0_uav = stats_p0['per_uav'][i]
    p1_uav = stats_p1['per_uav'][i]

    print(f"{'悬停点数':<20} {r_uav['num_hover']:<15} {f_uav['num_hover']:<15} {p0_uav['num_hover']:<15} {p1_uav['num_hover']:<15}")
    print(f"{'服务设备数':<20} {r_uav['num_devices']:<15} {f_uav['num_devices']:<15} {p0_uav['num_devices']:<15} {p1_uav['num_devices']:<15}")
    print(f"{'飞行距离 (m)':<20} {r_uav['distance']:<15.2f} {f_uav['distance']:<15.2f} {p0_uav['distance']:<15.2f} {p1_uav['distance']:<15.2f}")
    print(f"{'飞行能耗 (J)':<20} {r_uav['energy']:<15.2f} {f_uav['energy']:<15.2f} {p0_uav['energy']:<15.2f} {p1_uav['energy']:<15.2f}")
    print(f"{'服务效率':<20} {r_uav['efficiency']:<15.2f} {f_uav['efficiency']:<15.2f} {p0_uav['efficiency']:<15.2f} {p1_uav['efficiency']:<15.2f}")

# ============================================================================
# 5. 可视化对比
# ============================================================================
print('\n[Step 4] 生成可视化对比图...')

def plot_trajectory_comparison(results_list, stats_list, iot_positions, save_path=None):
    """绘制四种方法的航迹对比（2×2布局）"""

    # 使用2×2布局的图形尺寸
    fig = plt.figure(figsize=(16, 14), dpi=100)

    # 论文级配色方案
    methods = ['Rand-NOMA', 'Max-Hover', 'Joint-Opt', 'Adam-2opt']
    method_colors = {
        'Rand-NOMA': '#9b59b6',      # 紫色
        'Max-Hover': '#95a5a6',      # 灰色
        'Joint-Opt': '#3498db',            # 蓝色
        'Adam-2opt': '#e74c3c'  # 红色（突出）
    }

    # UAV轨迹颜色（高对比度配色，易于区分）
    uav_colors = ['#E63946', '#06A77D', '#F77F00']  # 鲜红、深绿、橙色
    uav_styles = ['-', '-', '-']  # 全部使用实线

    # IoT设备和悬停点配色
    iot_paired_color = '#3498db'    # 蓝色
    iot_unpaired_color = '#e74c3c'  # 红色
    hover_color = '#2ecc71'         # 绿色
    start_color = '#f39c12'         # 橙色

    # 航迹可视化（4个子图，2行2列）
    for idx, (result, stats, method) in enumerate(zip(results_list, stats_list, methods)):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(2, 2, idx + 1)

        # 绘制IoT设备
        paired_devices = set()
        for k, m in result['paired']:
            paired_devices.add(k)
            paired_devices.add(m)

        unpaired_devices = result['unpaired']

        # 配对设备（蓝色圆点，更大更清晰）
        if len(paired_devices) > 0:
            paired_indices = list(paired_devices)
            ax.scatter(iot_positions[paired_indices, 0], iot_positions[paired_indices, 1],
                      c=iot_paired_color, marker='o', s=100, label='Paired IoT',
                      alpha=0.7, edgecolors='white', linewidths=1.5, zorder=2)

        # 未配对设备（红色方块）
        if len(unpaired_devices) > 0:
            ax.scatter(iot_positions[unpaired_devices, 0], iot_positions[unpaired_devices, 1],
                      c=iot_unpaired_color, marker='s', s=100, label='Unpaired IoT',
                      alpha=0.7, edgecolors='white', linewidths=1.5, zorder=2)

        # 绘制配对连线（更细更淡）
        for k, m in result['paired']:
            ax.plot([iot_positions[k, 0], iot_positions[m, 0]],
                   [iot_positions[k, 1], iot_positions[m, 1]],
                   color=iot_paired_color, linestyle='--', alpha=0.15, linewidth=1, zorder=1)

        # 绘制悬停点（更突出）
        hover_pos = result['hover_positions']
        ax.scatter(hover_pos[:, 0], hover_pos[:, 1],
                  c=hover_color, marker='^', s=180, label='Hover Point',
                  edgecolors='darkgreen', linewidths=2, zorder=4, alpha=0.9)

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

            # 轨迹线（更粗更清晰）
            ax.plot(path_x, path_y, linestyle=uav_styles[u], color=uav_colors[u],
                   linewidth=2.5, label=f'UAV {u+1}', alpha=0.8, zorder=3)

            # 添加方向箭头（更精致）
            for i in range(len(path_x) - 1):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 50:  # 只在较长的线段上添加箭头
                    mid_x = path_x[i] + dx * 0.5
                    mid_y = path_y[i] + dy * 0.5
                    ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1),
                               xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                               arrowprops=dict(arrowstyle='->', color=uav_colors[u],
                                             lw=2, alpha=0.6), zorder=2)

        # 起点/终点（更醒目）
        ax.scatter(start_pos[0], start_pos[1], c=start_color, marker='*',
                  s=500, label='Base Station', edgecolors='darkorange',
                  linewidths=2.5, zorder=5, alpha=1.0)

        # 标题 - 显示关键指标
        uav_energy = stats["hover_energy"] + stats["flight_energy"]
        hover_points = stats["num_paired"] + stats["num_unpaired"]

        # 根据算法类型使用不同的边框颜色
        title_color = method_colors[method]

        # 添加子图编号
        subfig_label = ['(a)', '(b)', '(c)', '(d)'][idx]
        title_text = f'{method}\n'
        title_text += f'Distance: {stats["flight_distance"]:.0f}m | '
        title_text += f'Energy: {uav_energy:.0f}J\n'
        title_text += f'Hover Points: {hover_points} | Paired: {stats["num_paired"]}'

        ax.set_title(title_text, fontsize=12,
                    color=title_color, pad=12)

        # 坐标轴标签 - 将子图编号放在X轴下方
        ax.set_xlabel(f'X (m)\n{subfig_label} {method}', fontsize=13)
        if col == 0:  # 只在左侧列显示Y轴标签
            ax.set_ylabel('Y (m)', fontsize=13)

        # 网格和边框
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.set_aspect('equal')
        ax.set_xlim(-30, Config.AREA_SIZE + 30)
        ax.set_ylim(-30, Config.AREA_SIZE + 30)
        ax.tick_params(direction='in', which='both')

        # 为Adam-2opt算法添加边框高亮
        if method == 'Adam-2opt':
            for spine in ax.spines.values():
                spine.set_edgecolor(title_color)
                spine.set_linewidth(3)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(1)

    # 创建统一的图例（放在2×2布局下方）
    handles, labels = fig.axes[0].get_legend_handles_labels()

    # 调整图例顺序和样式 - 符合SCI论文标准
    legend = fig.legend(handles, labels,
                       loc='lower center',
                       bbox_to_anchor=(0.5, -0.02),
                       ncol=7,  # 水平排列
                       fontsize=11,
                       frameon=True,
                       fancybox=False,  # 不使用圆角
                       shadow=False,    # 不使用阴影
                       framealpha=0.95,
                       edgecolor='black',  # 黑色边框
                       borderpad=0.8)

    # 调整布局（为底部图例留出空间）
    plt.tight_layout(rect=[0, 0.04, 1, 1.0])

    if save_path:
        # 保存PNG格式
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'\n可视化图已保存到: {save_path}')

        # 保存EPS格式（矢量图，适合论文发表）
        eps_path = save_path.replace('.png', '.eps')
        plt.savefig(eps_path, format='eps', bbox_inches='tight', facecolor='white')
        print(f'EPS格式已保存到: {eps_path}')

    plt.show()  # 显示图片窗口

# 生成可视化
# 确保使用绝对路径并创建目录
script_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(script_dir, '..', '结果图表')
os.makedirs(result_dir, exist_ok=True)
save_path = os.path.abspath(os.path.join(result_dir, 'trajectory_comparison_4algorithms.png'))

plot_trajectory_comparison(
    [result_rscheme, result_fscheme, result_p0, result_p1],
    [stats_rscheme, stats_fscheme, stats_p0, stats_p1],
    iot_positions,
    save_path=save_path
)

print('\n' + '='*80)
print('航迹诊断与可视化完成！')
print('='*80)
