"""
基线方法实现 (Baseline Methods)
用于对比实验，验证论文方法的优越性

优化目标: 最小化UAV能耗 (E_hover + E_flight)
- E_hover: UAV悬停能耗（数据采集期间）
- E_flight: UAV飞行能耗（轨迹移动期间）

包含:
- R-scheme: 随机配对 + 优化悬停点
- F-scheme: 固定悬停（无NOMA，无优化）
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict
from algorithm1 import (
    Config, channel_gain, calculate_sinr, calculate_rate,
    compute_transmission_energy, optimize_uav_position,
    nearest_neighbor_tsp, compute_trajectory_length,
    optimize_power_pair
)


# ============================================================================
# Baseline 1: R-scheme (Random Pairing)
# ============================================================================

def random_pairing(iot_positions: np.ndarray,
                   distance_threshold: float = Config.DISTANCE_THRESHOLD,
                   seed: int = None) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    R-scheme: 随机配对算法

    特点:
    - IoT设备随机配对（而非贪心最近邻）
    - 保留距离阈值约束
    - 无配对交换优化

    Args:
        iot_positions: IoT位置 [K, 3]
        distance_threshold: 配对最大距离
        seed: 随机种子

    Returns:
        paired: 配对列表
        unpaired: 未配对列表
    """
    if seed is not None:
        np.random.seed(seed)

    K = len(iot_positions)
    dist_matrix = cdist(iot_positions, iot_positions)

    paired = []
    unpaired = []
    available = list(range(K))

    # 随机打乱顺序
    np.random.shuffle(available)

    while len(available) >= 2:
        # 随机选择第一个设备
        i = available[0]

        # 随机选择配对候选（满足距离约束）
        valid_candidates = [j for j in available[1:]
                           if dist_matrix[i, j] < distance_threshold]

        if len(valid_candidates) > 0:
            # 随机选择一个配对
            j = np.random.choice(valid_candidates)
            paired.append((i, j))
            available.remove(i)
            available.remove(j)
        else:
            # 无法配对
            unpaired.append(i)
            available.remove(i)

    # 剩余设备
    unpaired.extend(available)

    return paired, unpaired


def r_scheme(iot_positions: np.ndarray,
            iot_data_sizes: np.ndarray,
            num_uavs: int = Config.NUM_UAVS,
            seed: int = None,
            verbose: bool = False) -> Dict:
    """
    R-scheme 完整方法

    流程:
    1. 随机配对（保留距离约束）
    2. 优化功率
    3. 优化悬停位置
    4. 轨迹规划
    5. 无配对交换优化

    Returns:
        result: 与algorithm_1相同格式的结果字典
    """
    if verbose:
        print("\n" + "="*60)
        print("R-scheme: Random Pairing")
        print("="*60)

    K = len(iot_positions)

    # Step 1: 随机配对
    if verbose:
        print("\n[Step 1] Random NOMA Pairing...")

    paired, unpaired = random_pairing(iot_positions, seed=seed)

    if verbose:
        print(f"  - Paired groups: {len(paired)}")
        print(f"  - Unpaired nodes: {len(unpaired)}")

    # Step 2-4: 与algorithm_1相同的优化流程
    hover_positions = []
    powers = {}

    # 处理配对节点
    for k, m in paired:
        pos_k = iot_positions[k]
        pos_m = iot_positions[m]
        data_k = iot_data_sizes[k]
        data_m = iot_data_sizes[m]

        # 初始UAV位置
        uav_init = (pos_k + pos_m) / 2
        uav_init[2] = Config.H_U

        # 优化功率和位置（简化：只迭代10次）
        for _ in range(10):
            pk, pm = optimize_power_pair(pos_k, pos_m, uav_init, data_k, data_m)
            uav_optimized = optimize_uav_position(
                [pos_k, pos_m], [data_k, data_m], [pk, pm],
                uav_init, is_noma_pair=True
            )

            if np.linalg.norm(uav_optimized - uav_init) < Config.CONVERGENCE_TOL:
                break
            uav_init = uav_optimized

        hover_positions.append(uav_optimized)
        powers[k] = pk
        powers[m] = pm

    # 处理未配对节点
    for idx in unpaired:
        pos = iot_positions[idx].copy()
        pos[2] = Config.H_U
        hover_positions.append(pos)
        powers[idx] = Config.P_MAX

    hover_positions = np.array(hover_positions)

    # Step 3: 轨迹规划
    trajectories = nearest_neighbor_tsp(hover_positions, num_uavs)

    # Step 4: 计算能耗
    energy_dict = compute_total_energy_baseline(
        paired, unpaired, powers, iot_positions, iot_data_sizes,
        hover_positions, trajectories
    )

    if verbose:
        print(f"\n[Result] Energy Breakdown:")
        print(f"  - UAV Total Energy: {energy_dict['uav_energy']:.2f} J ← [PRIMARY METRIC]")
        print("="*60)

    return {
        'paired': paired,
        'unpaired': unpaired,
        'powers': powers,
        'hover_positions': hover_positions,
        'trajectories': trajectories,
        'energy': energy_dict,
        'uav_energy': energy_dict['uav_energy'],      # UAV能耗 (优化目标)
        'method': 'R-scheme'
    }


# ============================================================================
# Baseline 2: F-scheme (Fixed Hovering)
# ============================================================================

def f_scheme(iot_positions: np.ndarray,
            iot_data_sizes: np.ndarray,
            num_uavs: int = Config.NUM_UAVS,
            verbose: bool = False) -> Dict:
    """
    F-scheme: 固定悬停方案

    特点:
    - 所有IoT独立上传（OFDMA，无NOMA）
    - UAV直接飞到每个IoT正上方悬停
    - 无功率优化（使用最大功率）
    - 无悬停点优化

    Returns:
        result: 与algorithm_1相同格式的结果字典
    """
    if verbose:
        print("\n" + "="*60)
        print("F-scheme: Fixed Hovering (No NOMA)")
        print("="*60)

    K = len(iot_positions)

    # 无配对，所有设备独立
    paired = []
    unpaired = list(range(K))

    if verbose:
        print(f"\n[Step 1] No Pairing (OFDMA only)")
        print(f"  - All {K} devices use OFDMA")

    # 固定悬停：直接在每个IoT上方
    hover_positions = []
    powers = {}

    for idx in range(K):
        pos = iot_positions[idx].copy()
        pos[2] = Config.H_U  # 固定高度
        hover_positions.append(pos)
        powers[idx] = Config.P_MAX  # 使用最大功率

    hover_positions = np.array(hover_positions)

    if verbose:
        print(f"\n[Step 2] Fixed Hovering Positions")
        print(f"  - {K} hover points (directly above each IoT)")

    # 轨迹规划
    trajectories = nearest_neighbor_tsp(hover_positions, num_uavs)

    if verbose:
        print(f"\n[Step 3] Trajectory Planning")
        for u, traj in enumerate(trajectories):
            print(f"  - UAV {u+1}: visits {len(traj)} points")

    # 计算能耗
    energy_dict = compute_total_energy_baseline(
        paired, unpaired, powers, iot_positions, iot_data_sizes,
        hover_positions, trajectories
    )

    if verbose:
        print(f"\n[Result] Energy Breakdown:")
        print(f"  - UAV Total Energy: {energy_dict['uav_energy']:.2f} J ← [PRIMARY METRIC]")
        print("="*60)

    return {
        'paired': paired,
        'unpaired': unpaired,
        'powers': powers,
        'hover_positions': hover_positions,
        'trajectories': trajectories,
        'energy': energy_dict,
        'uav_energy': energy_dict['uav_energy'],      # UAV能耗 (优化目标)
        'method': 'F-scheme'
    }


# ============================================================================
# 能耗计算（统一接口）
# ============================================================================

def compute_total_energy_baseline(paired: List[Tuple[int, int]],
                                  unpaired: List[int],
                                  powers: Dict[int, float],
                                  iot_positions: np.ndarray,
                                  iot_data_sizes: np.ndarray,
                                  hover_positions: np.ndarray,
                                  trajectories: List[List[int]]) -> Dict:
    """
    统一的能耗计算函数（适用于所有方法）

    Returns:
        {
            'hover_energy': UAV悬停能耗,
            'flight_energy': UAV飞行能耗,
            'uav_energy': UAV总能耗 (优化目标: hover + flight),
            'hover_time': 总悬停时间,
            'flight_distance': 总飞行距离
        }
    """
    # 1. 计算总悬停时间
    T_hover_total = 0

    # NOMA配对
    for (k, m) in paired:
        pk = powers[k]
        pm = powers[m]

        pair_idx = paired.index((k, m))
        if pair_idx < len(hover_positions):
            uav_pos = hover_positions[pair_idx]
        else:
            uav_pos = np.array([Config.AREA_SIZE/2, Config.AREA_SIZE/2, Config.H_U])

        Gk = channel_gain(iot_positions[k], uav_pos)
        Gm = channel_gain(iot_positions[m], uav_pos)

        # 设备k的传输时间
        sinr_k = calculate_sinr(pk, pm, Gk, Gm, is_paired=True)
        rate_k = calculate_rate(sinr_k)
        T_k = iot_data_sizes[k] / rate_k

        # 设备m的传输时间
        sinr_m = calculate_sinr(pm, 0, Gm, 0, is_paired=False)
        rate_m = calculate_rate(sinr_m)
        T_m = iot_data_sizes[m] / rate_m

        T_hover_total += (T_k + T_m)

    # OFDMA独立节点
    for idx in unpaired:
        p = powers[idx]

        unpaired_hover_idx = len(paired) + unpaired.index(idx)
        if unpaired_hover_idx < len(hover_positions):
            uav_pos = hover_positions[unpaired_hover_idx]
        else:
            uav_pos = iot_positions[idx].copy()
            uav_pos[2] = Config.H_U

        G = channel_gain(iot_positions[idx], uav_pos)
        sinr = calculate_sinr(p, 0, G, 0, is_paired=False)
        rate = calculate_rate(sinr)
        T = iot_data_sizes[idx] / rate

        T_hover_total += T

    # 2. UAV悬停能耗
    E_hover = Config.P_H * T_hover_total

    # 3. UAV飞行能耗
    E_flight = 0
    total_flight_distance = 0
    start_pos = np.array([Config.AREA_SIZE/2, Config.AREA_SIZE/2, Config.H_U])

    for trajectory in trajectories:
        length = compute_trajectory_length(trajectory, hover_positions, start_pos)
        total_flight_distance += length
        T_fly = length / Config.V_F
        E_flight += Config.P_F * T_fly

    # 4. UAV总能耗 (优化目标)
    E_uav = E_hover + E_flight

    return {
        'hover_energy': E_hover,          # UAV悬停能耗
        'flight_energy': E_flight,        # UAV飞行能耗
        'uav_energy': E_uav,              # UAV总能耗 (优化目标)
        'hover_time': T_hover_total,
        'flight_distance': total_flight_distance
    }
