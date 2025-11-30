"""
IoT-UAV Phase Optimization (Algorithm 1) - P0 Base Version
Paper: Joint UAV Trajectory Planning and LEO Satellite Selection

Optimization Objective: Minimize UAV Energy (E_hover + E_flight)
- E_hover: UAV hovering energy during data collection
- E_flight: UAV flight energy for trajectory movement
- Note: IoT energy is calculated but NOT included in optimization objective

This is the P0 base version (without Adam and 2-opt optimizations)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 全局参数配置 (Table I)
# ============================================================================

class Config:
    """仿真参数配置"""
    # 功率参数
    P_MAX = 5.0          # W - IoT最大发射功率
    P_MIN = 0.1          # W - IoT最小发射功率
    P_H = 80.0           # W - UAV悬停功率
    P_F = 240.0          # W - UAV飞行功率

    # 信道参数
    BETA_0 = 9.89e-5     # 参考距离信道增益
    B_IU = 1e6           # Hz - IoT-UAV带宽 (1 MHz)
    SIGMA_IU_SQ = 1e-18  # W - 噪声功率

    # NOMA参数
    RHO = 0.8             # NOMA功率差系数
    MIN_GAIN_RATIO = 1.05 # 最小增益比（防止等距配对性能退化）
                          # 注：由于H_U=200m远大于DISTANCE_THRESHOLD=80m，
                          # 可实现的最大增益比约为1.16，因此设为1.05较合理

    # UAV参数
    H_U = 200.0          # m - UAV固定飞行高度
    V_F = 20.0           # m/s - UAV飞行速度 (论文未给出，假设)
    NUM_UAVS = 3         # UAV数量

    # 场景参数
    AREA_SIZE = 500.0    # m - 区域边长

    # 算法参数
    MAX_ITER = 100       # 最大迭代次数
    CONVERGENCE_TOL = 1e-4  # 收敛阈值
    DISTANCE_THRESHOLD = 80.0  # m - NOMA配对最大距离
    BOUNDARY_MARGIN = 50.0  # m - 边界缓冲区宽度（防止悬停点在极端边界）

    # 数据参数 (论文未明确给出)
    DATA_SIZE_MIN = 100e3 * 8   # bits - 最小数据包 (100KB)
    DATA_SIZE_MAX = 1e6 * 8     # bits - 最大数据包 (1MB)

# ============================================================================
# 2. 核心计算函数
# ============================================================================

def channel_gain(pos_iot: np.ndarray, pos_uav: np.ndarray) -> float:
    """
    计算IoT到UAV的信道增益 (公式1)
    
    Args:
        pos_iot: IoT位置 [x, y, 0]
        pos_uav: UAV位置 [x, y, h_u]
    
    Returns:
        信道增益 G_k
    """
    distance = np.linalg.norm(pos_iot - pos_uav)
    if distance < 1e-6:  # 避免除零
        distance = 1e-6
    G = Config.BETA_0 / (distance ** 2)
    return G


def calculate_sinr(pk: float, pm: float, Gk: float, Gm: float, 
                   is_paired: bool = False) -> float:
    """
    计算SINR (公式2)
    
    Args:
        pk: 设备k的发射功率
        pm: 设备m的发射功率 (配对情况)
        Gk: 设备k的信道增益
        Gm: 设备m的信道增益
        is_paired: 是否为NOMA配对
    
    Returns:
        SINR值
    """
    if is_paired and pm > 0:
        interference = pm * Gm
    else:
        interference = 0
    
    sinr = pk * Gk / (interference + Config.SIGMA_IU_SQ)
    return sinr


def calculate_rate(sinr: float) -> float:
    """
    计算数据传输速率 (公式3)
    
    Args:
        sinr: 信干噪比
    
    Returns:
        数据速率 (bps)
    """
    rate = Config.B_IU * np.log2(1 + sinr)
    return rate


def compute_transmission_energy(power: float, data_size: float, 
                                rate: float) -> float:
    """
    计算传输能耗 (公式5)
    
    Args:
        power: 发射功率 (W)
        data_size: 数据量 (bits)
        rate: 传输速率 (bps)
    
    Returns:
        能耗 (J)
    """
    if rate < 1e-6:  # 避免除零
        return 1e10  # 惩罚值
    transmission_time = data_size / rate
    energy = power * transmission_time
    return energy

# ============================================================================
# 3. IoT设备配对算法
# ============================================================================

def initial_pairing(iot_positions: np.ndarray,
                   distance_threshold: float = Config.DISTANCE_THRESHOLD,
                   min_gain_ratio: float = Config.MIN_GAIN_RATIO) \
                   -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    贪心配对算法: 最近邻优先 + 增益比筛选 (Algorithm 1, Lines 1-2)

    Args:
        iot_positions: IoT设备位置数组 [K, 3]
        distance_threshold: 配对最大距离
        min_gain_ratio: 最小增益比（防止等距配对性能退化）

    Returns:
        paired: 配对列表 [(k1, m1), (k2, m2), ...]
        unpaired: 未配对设备索引列表

    改进说明:
        - 添加增益比筛选：确保 Gk/Gm >= min_gain_ratio
        - 防止等距或近等距配对导致NOMA性能退化
        - 估计UAV位置为两设备中点上方
    """
    K = len(iot_positions)
    dist_matrix = cdist(iot_positions, iot_positions)
    np.fill_diagonal(dist_matrix, np.inf)

    paired = []
    unpaired = []
    available = list(range(K))  # 可配对的候选集

    # 贪心配对
    while len(available) >= 2:
        # 选择第一个可用设备
        i = available[0]

        # 找到最近的可用设备
        neighbors = [(dist_matrix[i, j], j) for j in available if j != i]

        if len(neighbors) == 0:
            # 没有其他可用设备，标记为未配对
            unpaired.append(i)
            available.remove(i)
            continue

        min_dist, j = min(neighbors, key=lambda x: x[0])

        # 检查距离约束
        if min_dist >= distance_threshold:
            # 距离太远，无法配对
            unpaired.append(i)
            available.remove(i)
            continue

        # 检查增益比约束
        # 策略：尝试将UAV放在设备i正上方，检查增益比
        # 如果设备i和j距离足够近，可以从i上方服务两者
        # 但需要确保增益差异足够大以支持NOMA

        pos_i = iot_positions[i]
        pos_j = iot_positions[j]

        # 尝试UAV在设备i正上方
        uav_above_i = np.array([pos_i[0], pos_i[1], Config.H_U])

        # 计算两设备到UAV的距离
        dist_i_from_above_i = np.linalg.norm(pos_i - uav_above_i)  # 应该≈H_U
        dist_j_from_above_i = np.linalg.norm(pos_j - uav_above_i)

        # 尝试UAV在设备j正上方
        uav_above_j = np.array([pos_j[0], pos_j[1], Config.H_U])
        dist_i_from_above_j = np.linalg.norm(pos_i - uav_above_j)
        dist_j_from_above_j = np.linalg.norm(pos_j - uav_above_j)  # 应该≈H_U

        # 选择增益比更好的位置
        # 位置1: UAV在i上方，i是近端，j是远端
        Gi_1 = channel_gain(pos_i, uav_above_i)
        Gj_1 = channel_gain(pos_j, uav_above_i)
        gain_ratio_1 = Gi_1 / Gj_1 if Gj_1 > 0 else 1.0

        # 位置2: UAV在j上方，j是近端，i是远端
        Gi_2 = channel_gain(pos_i, uav_above_j)
        Gj_2 = channel_gain(pos_j, uav_above_j)
        gain_ratio_2 = Gj_2 / Gi_2 if Gi_2 > 0 else 1.0

        # 选择增益比更大的方案
        best_gain_ratio = max(gain_ratio_1, gain_ratio_2)

        if best_gain_ratio < min_gain_ratio:
            # 即使最优位置也无法满足增益比要求，拒绝配对
            unpaired.append(i)
            available.remove(i)
            continue

        # 改进的配对策略: 在满足NOMA约束的前提下，选择能耗最小的方案
        # 评估两种配对方案的能耗
        candidates = []

        # 方案1: UAV在i上方，i是近端，j是远端
        if gain_ratio_1 >= min_gain_ratio:
            # 估算能耗：使用距离作为能耗的代理指标（距离越短，能耗越低）
            total_dist_1 = dist_i_from_above_i + dist_j_from_above_i
            candidates.append((total_dist_1, i, j, gain_ratio_1))

        # 方案2: UAV在j上方，j是近端，i是远端
        if gain_ratio_2 >= min_gain_ratio:
            total_dist_2 = dist_i_from_above_j + dist_j_from_above_j
            candidates.append((total_dist_2, j, i, gain_ratio_2))

        if len(candidates) == 0:
            # 没有方案满足NOMA约束
            unpaired.append(i)
            available.remove(i)
            continue

        # 选择总距离最小的方案（能耗最优）
        candidates.sort(key=lambda x: x[0])
        _, k_idx, m_idx, selected_gain_ratio = candidates[0]

        # 通过所有检查，配对成功
        paired.append((k_idx, m_idx))  # 确保近端在前
        available.remove(i)
        available.remove(j)

    # available中剩余的单个设备也是unpaired
    unpaired.extend(available)

    return paired, unpaired


# ============================================================================
# 4. 功率优化 (牛顿法)
# ============================================================================

def optimize_power_pair(pos_k: np.ndarray, pos_m: np.ndarray, 
                       pos_uav: np.ndarray, data_k: float, data_m: float,
                       max_iter: int = 50) -> Tuple[float, float]:
    """
    交替优化NOMA配对的功率 (Algorithm 1, Line 5)
    使用梯度下降法 (简化版牛顿法)
    
    Args:
        pos_k, pos_m: 设备k和m的位置
        pos_uav: UAV位置
        data_k, data_m: 数据量
        max_iter: 最大迭代次数
    
    Returns:
        优化后的 (pk, pm)
    """
    # 计算信道增益
    Gk = channel_gain(pos_k, pos_uav)
    Gm = channel_gain(pos_m, pos_uav)
    
    # 初始化功率
    pk = (Config.P_MIN + Config.P_MAX) / 2
    pm = (Config.P_MIN + Config.P_MAX) / 2
    
    learning_rate = 0.01
    
    for iteration in range(max_iter):
        pk_old, pm_old = pk, pm
        
        # 固定pm，优化pk
        # 目标: min E_k = pk * T_k = pk * Dk / rate_k
        sinr_k = calculate_sinr(pk, pm, Gk, Gm, is_paired=True)
        rate_k = calculate_rate(sinr_k)
        
        # 数值梯度 (避免复杂的解析推导)
        delta = 1e-6
        sinr_k_plus = calculate_sinr(pk + delta, pm, Gk, Gm, is_paired=True)
        rate_k_plus = calculate_rate(sinr_k_plus)
        
        energy_k = pk * data_k / (rate_k + 1e-10)
        energy_k_plus = (pk + delta) * data_k / (rate_k_plus + 1e-10)
        
        grad_pk = (energy_k_plus - energy_k) / delta
        pk = pk - learning_rate * grad_pk
        
        # 投影到可行域
        pk = np.clip(pk, Config.P_MIN, Config.P_MAX)
        
        # 固定pk，优化pm (同样过程)
        sinr_m = calculate_sinr(pm, 0, Gm, 0, is_paired=False)  # m的信号后解码
        rate_m = calculate_rate(sinr_m)
        
        sinr_m_plus = calculate_sinr(pm + delta, 0, Gm, 0, is_paired=False)
        rate_m_plus = calculate_rate(sinr_m_plus)
        
        energy_m = pm * data_m / (rate_m + 1e-10)
        energy_m_plus = (pm + delta) * data_m / (rate_m_plus + 1e-10)
        
        grad_pm = (energy_m_plus - energy_m) / delta
        pm = pm - learning_rate * grad_pm

        # 投影到可行域
        pm = np.clip(pm, Config.P_MIN, Config.P_MAX)

        # NOMA约束: 远端功率应该大于近端功率，确保SIC正常工作
        # pm是远端用户(信道差), pk是近端用户(信道好)
        pm = max(pm, 1.2 * pk)
        pm = np.clip(pm, Config.P_MIN, Config.P_MAX)
        
        # 检查收敛
        if abs(pk - pk_old) < 1e-5 and abs(pm - pm_old) < 1e-5:
            break
    
    return pk, pm


# ============================================================================
# 5. UAV位置优化 (Nelder-Mead)
# ============================================================================

def optimize_uav_position(iot_positions: List[np.ndarray],
                         iot_data_sizes: List[float],
                         powers: List[float],
                         initial_pos: np.ndarray,
                         is_noma_pair: bool = True,
                         use_boundary_margin: bool = True) -> np.ndarray:
    """
    优化UAV悬停位置 (Algorithm 1, Line 6)

    Args:
        iot_positions: IoT设备位置列表
        iot_data_sizes: 数据量列表
        powers: 功率列表
        initial_pos: 初始UAV位置
        is_noma_pair: 是否为NOMA配对
        use_boundary_margin: 是否使用边界缓冲区约束（防止极端边界位置）

    Returns:
        优化后的UAV位置 [x, y, h_u]
    """
    def objective(uav_xy):
        """目标函数: 悬停能耗"""
        uav_pos = np.array([uav_xy[0], uav_xy[1], Config.H_U])

        total_energy = 0

        if is_noma_pair and len(iot_positions) == 2:
            # NOMA配对情况
            pos_k, pos_m = iot_positions
            pk, pm = powers
            data_k, data_m = iot_data_sizes

            Gk = channel_gain(pos_k, uav_pos)
            Gm = channel_gain(pos_m, uav_pos)

            # 设备k的能耗
            sinr_k = calculate_sinr(pk, pm, Gk, Gm, is_paired=True)
            rate_k = calculate_rate(sinr_k)
            energy_k = compute_transmission_energy(pk, data_k, rate_k)

            # 设备m的能耗
            sinr_m = calculate_sinr(pm, 0, Gm, 0, is_paired=False)
            rate_m = calculate_rate(sinr_m)
            energy_m = compute_transmission_energy(pm, data_m, rate_m)

            total_energy = energy_k + energy_m
        else:
            # OFDMA独立节点
            for pos, p, data in zip(iot_positions, powers, iot_data_sizes):
                G = channel_gain(pos, uav_pos)
                sinr = calculate_sinr(p, 0, G, 0, is_paired=False)
                rate = calculate_rate(sinr)
                energy = compute_transmission_energy(p, data, rate)
                total_energy += energy

        return total_energy

    # 设置边界约束
    if use_boundary_margin:
        # 使用边界缓冲区：[margin, AREA_SIZE-margin]
        margin = Config.BOUNDARY_MARGIN
        bounds = [(margin, Config.AREA_SIZE - margin),
                  (margin, Config.AREA_SIZE - margin)]
    else:
        # 原始边界：[0, AREA_SIZE]
        bounds = [(0, Config.AREA_SIZE), (0, Config.AREA_SIZE)]

    # 使用L-BFGS-B优化（支持边界约束）
    result = minimize(
        objective,
        x0=initial_pos[:2],
        method='L-BFGS-B',
        bounds=bounds,  # 应用边界约束
        options={'ftol': 1e-6, 'maxiter': 200}
    )

    optimized_pos = np.array([result.x[0], result.x[1], Config.H_U])
    return optimized_pos


# ============================================================================
# 6. UAV轨迹规划 (简化版 - 最近邻TSP)
# ============================================================================

def nearest_neighbor_tsp(hover_points: np.ndarray, 
                        num_uavs: int = Config.NUM_UAVS,
                        start_pos: np.ndarray = None) -> List[List[int]]:
    """
    简化的TSP求解: 最近邻启发式 (Algorithm 1, Line 10)
    论文使用GWO，这里用简单方法替代
    
    Args:
        hover_points: 所有悬停点 [N, 3]
        num_uavs: UAV数量
        start_pos: 起始位置
    
    Returns:
        trajectories: 每个UAV的访问顺序列表
    """
    N = len(hover_points)
    
    if start_pos is None:
        start_pos = np.array([Config.AREA_SIZE/2, Config.AREA_SIZE/2, Config.H_U])
    
    # 简单分配: 每个UAV负责N/num_uavs个点
    points_per_uav = N // num_uavs
    trajectories = []
    
    unvisited = list(range(N))
    
    for u in range(num_uavs):
        trajectory = []
        current_pos = start_pos.copy()
        
        # 分配给该UAV的点数
        num_points = points_per_uav if u < num_uavs - 1 else len(unvisited)
        
        for _ in range(num_points):
            if len(unvisited) == 0:
                break
            
            # 找最近的未访问点
            distances = [np.linalg.norm(hover_points[i] - current_pos) 
                        for i in unvisited]
            nearest_idx = unvisited[np.argmin(distances)]
            
            trajectory.append(nearest_idx)
            unvisited.remove(nearest_idx)
            current_pos = hover_points[nearest_idx]
        
        trajectories.append(trajectory)
    
    return trajectories


def compute_trajectory_length(trajectory: List[int], 
                             hover_points: np.ndarray,
                             start_pos: np.ndarray) -> float:
    """
    计算轨迹总长度 (公式6)
    
    Args:
        trajectory: 访问顺序
        hover_points: 悬停点位置
        start_pos: 起始位置
    
    Returns:
        总飞行距离 (m)
    """
    if len(trajectory) == 0:
        return 0
    
    total_length = 0
    current_pos = start_pos
    
    for point_idx in trajectory:
        next_pos = hover_points[point_idx]
        total_length += np.linalg.norm(next_pos - current_pos)
        current_pos = next_pos
    
    # 返回起点
    total_length += np.linalg.norm(start_pos - current_pos)
    
    return total_length


# ============================================================================
# 7. 能耗计算
# ============================================================================

def compute_total_energy(paired: List[Tuple[int, int]],
                        unpaired: List[int],
                        powers: Dict[int, float],
                        iot_positions: np.ndarray,
                        iot_data_sizes: np.ndarray,
                        hover_positions: np.ndarray,
                        trajectories: List[List[int]]) -> Dict:
    """
    计算能耗 - 优化目标: UAV能耗 E^UAV = E_hover + E_flight

    Returns:
        能耗字典: {
            'hover_energy': UAV悬停能耗 (优化目标1),
            'flight_energy': UAV飞行能耗 (优化目标2),
            'uav_energy': UAV总能耗 (主要优化目标),
            'per_uav_stats': 每个UAV的详细统计
        }
    """
    # 1. 计算总悬停时间
    T_hover_total = 0

    # NOMA配对
    for (k, m) in paired:
        pk = powers[k]
        pm = powers[m]

        # 找到对应的UAV位置 (简化: 使用hover_positions中的对应点)
        # 这里假设hover_positions的顺序对应paired的顺序
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

        # 找对应的hover位置
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

    # 2. UAV悬停能耗 (公式8)
    E_hover = Config.P_H * T_hover_total

    # 3. UAV飞行能耗 (公式7) - 按UAV分别计算
    E_flight = 0
    total_flight_distance = 0
    start_pos = np.array([Config.AREA_SIZE/2, Config.AREA_SIZE/2, Config.H_U])

    # 每个UAV的详细统计
    per_uav_stats = []

    for uav_idx, trajectory in enumerate(trajectories):
        length = compute_trajectory_length(trajectory, hover_positions, start_pos)
        T_fly = length / Config.V_F
        E_fly = Config.P_F * T_fly

        # 统计该UAV服务的设备数
        num_hover_points = len(trajectory)

        # 统计服务的IoT设备数（NOMA配对算2个，OFDMA算1个）
        num_iot_devices = 0
        visited_hover_indices = trajectory
        for hover_idx in visited_hover_indices:
            if hover_idx < len(paired):
                num_iot_devices += 2  # NOMA配对
            else:
                num_iot_devices += 1  # OFDMA

        # 计算服务效率（设备数/悬停点数）
        service_efficiency = num_iot_devices / num_hover_points if num_hover_points > 0 else 0

        per_uav_stats.append({
            'uav_id': uav_idx + 1,
            'num_hover_points': num_hover_points,
            'num_iot_devices': num_iot_devices,
            'flight_distance': length,
            'flight_time': T_fly,
            'flight_energy': E_fly,
            'service_efficiency': service_efficiency,
            'trajectory': trajectory
        })

        total_flight_distance += length
        E_flight += E_fly

    # 4. UAV总能耗 (优化目标)
    E_uav = E_hover + E_flight

    return {
        'hover_energy': E_hover,          # UAV悬停能耗
        'flight_energy': E_flight,        # UAV飞行能耗
        'uav_energy': E_uav,              # UAV总能耗 (优化目标)
        'hover_time': T_hover_total,
        'flight_distance': total_flight_distance,
        'per_uav_stats': per_uav_stats
    }


# ============================================================================
# 7.5 能耗详细打印
# ============================================================================

def print_per_uav_statistics(energy_dict: Dict, method_name: str = "Proposed"):
    """
    打印每个UAV的详细统计信息

    Args:
        energy_dict: compute_total_energy的返回结果
        method_name: 方法名称（用于标题）
    """
    print(f"\n{'='*80}")
    print(f"Per-UAV Statistics - {method_name}")
    print(f"{'='*80}")

    per_uav_stats = energy_dict.get('per_uav_stats', [])

    if len(per_uav_stats) == 0:
        print("  No per-UAV statistics available.")
        return

    # 表头
    print(f"\n{'UAV':<6} {'Hover':<8} {'Devices':<9} {'Distance':<12} {'Time':<10} "
          f"{'Energy':<12} {'Efficiency':<10}")
    print(f"{'ID':<6} {'Points':<8} {'Served':<9} {'(m)':<12} {'(s)':<10} "
          f"{'(J)':<12} {'(dev/pt)':<10}")
    print("-" * 80)

    # 每个UAV的数据
    total_hover_points = 0
    total_devices = 0
    total_distance = 0
    total_flight_energy = 0

    for stats in per_uav_stats:
        uav_id = stats['uav_id']
        num_hover = stats['num_hover_points']
        num_devices = stats['num_iot_devices']
        distance = stats['flight_distance']
        time = stats['flight_time']
        energy = stats['flight_energy']
        efficiency = stats['service_efficiency']

        print(f"UAV{uav_id:<3} {num_hover:<8} {num_devices:<9} {distance:<12.2f} {time:<10.2f} "
              f"{energy:<12.2f} {efficiency:<10.2f}")

        total_hover_points += num_hover
        total_devices += num_devices
        total_distance += distance
        total_flight_energy += energy

    print("-" * 80)
    print(f"{'Total':<6} {total_hover_points:<8} {total_devices:<9} {total_distance:<12.2f} "
          f"{'-':<10} {total_flight_energy:<12.2f} {'-':<10}")

    # 负载均衡度分析
    distances = [s['flight_distance'] for s in per_uav_stats]
    energies = [s['flight_energy'] for s in per_uav_stats]

    if len(distances) > 1:
        dist_mean = np.mean(distances)
        dist_std = np.std(distances)
        dist_cv = dist_std / dist_mean if dist_mean > 0 else 0  # 变异系数

        print(f"\n[*] Load Balance Analysis:")
        print(f"   Flight Distance: Mean={dist_mean:.2f}m, Std={dist_std:.2f}m, CV={dist_cv:.3f}")
        print(f"   Flight Energy:   Mean={np.mean(energies):.2f}J, Std={np.std(energies):.2f}J")

        # 检测异常UAV
        threshold = dist_mean + 1.5 * dist_std
        abnormal_uavs = [s['uav_id'] for s in per_uav_stats if s['flight_distance'] > threshold]
        if len(abnormal_uavs) > 0:
            print(f"   [!] Abnormal UAVs (distance > {threshold:.2f}m): {abnormal_uavs}")

    print(f"{'='*80}\n")


def print_energy_comparison(result_proposed: Dict, result_fscheme: Dict):
    """
    对比Proposed和F-scheme的能耗

    Args:
        result_proposed: Proposed方法的结果
        result_fscheme: F-scheme的结果
    """
    print(f"\n{'='*80}")
    print(f"Energy Comparison: Proposed vs F-scheme")
    print(f"{'='*80}")

    # 总能耗对比
    energy_proposed = result_proposed['energy']
    energy_fscheme = result_fscheme['energy']

    print(f"\n[*] UAV Energy Consumption:")
    print(f"   {'Method':<15} {'Hover (J)':<12} {'Flight (J)':<12} {'UAV Total (J)':<15}")
    print("-" * 80)
    print(f"   {'Proposed':<15} {energy_proposed['hover_energy']:<12.2f} "
          f"{energy_proposed['flight_energy']:<12.2f} {energy_proposed['uav_energy']:<15.2f}")
    print(f"   {'F-scheme':<15} {energy_fscheme['hover_energy']:<12.2f} "
          f"{energy_fscheme['flight_energy']:<12.2f} {energy_fscheme['uav_energy']:<15.2f}")

    # 差异
    diff_hover = energy_proposed['hover_energy'] - energy_fscheme['hover_energy']
    diff_flight = energy_proposed['flight_energy'] - energy_fscheme['flight_energy']
    diff_uav = energy_proposed['uav_energy'] - energy_fscheme['uav_energy']

    print("-" * 80)
    print(f"   {'Difference':<15} {diff_hover:<12.2f} {diff_flight:<12.2f} {diff_uav:<15.2f}")
    print(f"   {'Percentage':<15} {diff_hover/energy_fscheme['hover_energy']*100:<11.2f}% "
          f"{diff_flight/energy_fscheme['flight_energy']*100:<11.2f}% "
          f"{diff_uav/energy_fscheme['uav_energy']*100:<11.2f}%")

    # Per-UAV对比
    print(f"\n[*] Per-UAV Comparison:")
    print(f"   {'UAV':<6} {'Method':<12} {'Distance (m)':<15} {'Energy (J)':<15} {'Devices':<10}")
    print("-" * 80)

    per_uav_proposed = energy_proposed.get('per_uav_stats', [])
    per_uav_fscheme = energy_fscheme.get('per_uav_stats', [])

    for i in range(max(len(per_uav_proposed), len(per_uav_fscheme))):
        if i < len(per_uav_proposed):
            p = per_uav_proposed[i]
            print(f"   UAV{p['uav_id']:<3} {'Proposed':<12} {p['flight_distance']:<15.2f} "
                  f"{p['flight_energy']:<15.2f} {p['num_iot_devices']:<10}")

        if i < len(per_uav_fscheme):
            f = per_uav_fscheme[i]
            print(f"   UAV{f['uav_id']:<3} {'F-scheme':<12} {f['flight_distance']:<15.2f} "
                  f"{f['flight_energy']:<15.2f} {f['num_iot_devices']:<10}")

        # 差异
        if i < len(per_uav_proposed) and i < len(per_uav_fscheme):
            p = per_uav_proposed[i]
            f = per_uav_fscheme[i]
            diff_dist = p['flight_distance'] - f['flight_distance']
            diff_energy = p['flight_energy'] - f['flight_energy']
            diff_dev = p['num_iot_devices'] - f['num_iot_devices']

            symbol_dist = "[OK]" if diff_dist <= 0 else "[!!]"
            symbol_energy = "[OK]" if diff_energy <= 0 else "[!!]"

            print(f"   {'   ':<6} {'Difference':<12} {diff_dist:<15.2f} {symbol_dist}  "
                  f"{diff_energy:<15.2f} {symbol_energy}  {diff_dev:<10}")

        if i < max(len(per_uav_proposed), len(per_uav_fscheme)) - 1:
            print()  # 空行分隔不同UAV

    print(f"{'='*80}\n")


# ============================================================================
# 8. 配对交换优化
# ============================================================================

def exchange_optimization(paired: List[Tuple[int, int]],
                         unpaired: List[int],
                         iot_positions: np.ndarray,
                         iot_data_sizes: np.ndarray,
                         current_energy: float) -> Tuple:
    """
    配对交换优化 (Algorithm 1, Lines 11-18)
    
    Args:
        paired: 当前配对
        unpaired: 当前未配对节点
        iot_positions: IoT位置
        iot_data_sizes: 数据量
        current_energy: 当前能耗
    
    Returns:
        (best_paired, best_unpaired, best_energy)
    """
    best_energy = current_energy
    best_paired = paired.copy()
    best_unpaired = unpaired.copy()
    
    # 策略1: 尝试从unpaired中形成新配对
    if len(unpaired) >= 2:
        for i in range(len(unpaired)):
            for j in range(i + 1, len(unpaired)):
                idx_i = unpaired[i]
                idx_j = unpaired[j]
                
                dist = np.linalg.norm(iot_positions[idx_i] - iot_positions[idx_j])
                
                if dist < Config.DISTANCE_THRESHOLD:
                    # 尝试配对
                    test_paired = paired + [(idx_i, idx_j)]
                    test_unpaired = [k for k in unpaired if k not in [idx_i, idx_j]]
                    
                    # 重新计算能耗 (简化版 - 只计算改变的部分)
                    # 完整版需要重新运行整个优化流程
                    # 这里使用启发式估计
                    improvement = estimate_pairing_benefit(
                        idx_i, idx_j, iot_positions, iot_data_sizes
                    )
                    
                    estimated_energy = current_energy - improvement
                    
                    if estimated_energy < best_energy:
                        best_energy = estimated_energy
                        best_paired = test_paired
                        best_unpaired = test_unpaired
    
    return best_paired, best_unpaired, best_energy


def estimate_pairing_benefit(idx_i: int, idx_j: int,
                             iot_positions: np.ndarray,
                             iot_data_sizes: np.ndarray) -> float:
    """
    估计配对带来的能耗减少 (启发式)
    
    Returns:
        能耗改进量 (正值表示减少)
    """
    # 简化估计: NOMA共享资源 vs OFDMA独立
    # 假设NOMA可以节约约20%的能耗
    pos_i = iot_positions[idx_i]
    pos_j = iot_positions[idx_j]
    
    # 计算中点位置
    mid_pos = (pos_i + pos_j) / 2
    mid_pos[2] = Config.H_U
    
    # 独立传输的能耗
    G_i = channel_gain(pos_i, mid_pos)
    G_j = channel_gain(pos_j, mid_pos)
    
    sinr_i = calculate_sinr(Config.P_MAX, 0, G_i, 0)
    sinr_j = calculate_sinr(Config.P_MAX, 0, G_j, 0)
    
    rate_i = calculate_rate(sinr_i)
    rate_j = calculate_rate(sinr_j)
    
    E_separate = (Config.P_MAX * iot_data_sizes[idx_i] / rate_i +
                  Config.P_MAX * iot_data_sizes[idx_j] / rate_j)
    
    # NOMA配对的能耗 (粗略估计)
    E_noma = E_separate * 0.8  # 假设节约20%
    
    benefit = E_separate - E_noma
    
    return max(0, benefit)


# ============================================================================
# 9. 主算法 - Algorithm 1
# ============================================================================

def algorithm_1(iot_positions: np.ndarray,
               iot_data_sizes: np.ndarray,
               num_uavs: int = Config.NUM_UAVS,
               use_gwo: bool = False,
               gwo_params: Dict = None,
               verbose: bool = True) -> Dict:
    """
    完整的Algorithm 1实现
    
    Args:
        iot_positions: IoT位置 [K, 3]
        iot_data_sizes: 数据量 [K]
        num_uavs: UAV数量
        use_gwo: 是否使用GWO轨迹规划（默认False使用最近邻）
        gwo_params: GWO参数字典 {'n_wolves': 30, 'max_iter': 100}
        verbose: 是否打印详细信息
    
    Returns:
        result: {
            'paired': 配对列表,
            'unpaired': 未配对列表,
            'powers': 功率字典,
            'hover_positions': 悬停位置,
            'trajectories': 轨迹,
            'energy': 能耗字典,
            'uav_energy': UAV总能耗 (优化目标),
            'total_energy': 系统总能耗 (参考)
        }

    优化目标: 最小化UAV能耗 (E_hover + E_flight)
    - 通过优化IoT功率减少传输时间 -> 降低E_hover
    - 通过优化UAV位置和轨迹 -> 降低E_flight
    """
    K = len(iot_positions)
    
    if verbose:
        print("="*60)
        print("Algorithm 1: IoT-UAV Phase Optimization")
        print("="*60)
        print(f"Number of IoT devices: {K}")
        print(f"Number of UAVs: {num_uavs}")
    
    # Step 1: 初始配对 (Lines 1-2)
    if verbose:
        print("\n[Step 1] Initial NOMA Pairing...")
    
    paired, unpaired = initial_pairing(iot_positions)
    
    if verbose:
        print(f"  - Paired groups: {len(paired)}")
        print(f"  - Unpaired nodes: {len(unpaired)}")
        print(f"  - Paired: {paired}")
        print(f"  - Unpaired: {unpaired}")
    
    # Step 2: 对每个配对优化功率和UAV位置 (Lines 2-9)
    if verbose:
        print("\n[Step 2] Power and Position Optimization...")
    
    hover_positions = []
    powers = {}
    
    # 处理配对节点
    for pair_idx, (k, m) in enumerate(paired):
        if verbose:
            print(f"  Processing pair {pair_idx + 1}/{len(paired)}: ({k}, {m})")
        
        pos_k = iot_positions[k]
        pos_m = iot_positions[m]
        data_k = iot_data_sizes[k]
        data_m = iot_data_sizes[m]
        
        # 初始UAV位置: 两设备中点
        uav_init = (pos_k + pos_m) / 2
        uav_init[2] = Config.H_U
        
        # 迭代优化
        for iteration in range(Config.MAX_ITER):
            # 优化功率 (Line 5)
            pk, pm = optimize_power_pair(pos_k, pos_m, uav_init, data_k, data_m)
            
            # 优化UAV位置 (Line 6)
            uav_optimized = optimize_uav_position(
                [pos_k, pos_m],
                [data_k, data_m],
                [pk, pm],
                uav_init,
                is_noma_pair=True
            )
            
            # 检查收敛
            if np.linalg.norm(uav_optimized - uav_init) < Config.CONVERGENCE_TOL:
                if verbose and iteration < 5:
                    print(f"    Converged at iteration {iteration + 1}")
                break
            
            uav_init = uav_optimized
        
        hover_positions.append(uav_optimized)
        powers[k] = pk
        powers[m] = pm
    
    # 处理未配对节点: 直接悬停在正上方，使用最大功率
    for idx in unpaired:
        pos = iot_positions[idx].copy()
        pos[2] = Config.H_U
        hover_positions.append(pos)
        powers[idx] = Config.P_MAX
    
    hover_positions = np.array(hover_positions)
    
    if verbose:
        print(f"  - Optimized {len(hover_positions)} hover positions")
    
    # Step 3: UAV轨迹规划 (Line 10)
    if verbose:
        method_name = "GWO" if use_gwo else "Nearest Neighbor"
        print(f"\n[Step 3] UAV Trajectory Planning ({method_name})...")

    if use_gwo:
        # 使用GWO优化轨迹
        try:
            from gwo_tsp import gwo_tsp

            # 设置GWO参数
            if gwo_params is None:
                gwo_params = {'n_wolves': 30, 'max_iter': 100}

            start_pos = np.array([Config.AREA_SIZE/2, Config.AREA_SIZE/2, Config.H_U])
            trajectories = gwo_tsp(
                hover_positions,
                num_uavs=num_uavs,
                start_pos=start_pos,
                n_wolves=gwo_params.get('n_wolves', 30),
                max_iter=gwo_params.get('max_iter', 100),
                verbose=verbose
            )
        except ImportError:
            if verbose:
                print("  警告: gwo_tsp模块未找到，回退到最近邻方法")
            trajectories = nearest_neighbor_tsp(hover_positions, num_uavs)
    else:
        # 使用最近邻启发式
        trajectories = nearest_neighbor_tsp(hover_positions, num_uavs)

    if verbose and not use_gwo:
        for u, traj in enumerate(trajectories):
            print(f"  - UAV {u + 1}: visits {len(traj)} points")
    
    # Step 4: 计算初始能耗
    if verbose:
        print("\n[Step 4] Computing Energy Consumption...")

    energy_dict = compute_total_energy(
        paired, unpaired, powers, iot_positions, iot_data_sizes,
        hover_positions, trajectories
    )

    if verbose:
        print(f"\n  [Optimization Objective: UAV Energy]")
        print(f"  - UAV Hovering Energy: {energy_dict['hover_energy']:.2f} J")
        print(f"  - UAV Flight Energy: {energy_dict['flight_energy']:.2f} J")
        print(f"  - UAV Total Energy: {energy_dict['uav_energy']:.2f} J ← [PRIMARY METRIC]")

        # 打印每个UAV的详细统计
        print_per_uav_statistics(energy_dict, method_name="Proposed")
    
    # Step 5: 配对交换优化 (Lines 11-18)
    if verbose:
        print("\n[Step 5] Pairing Exchange Optimization...")
    
    paired_new, unpaired_new, energy_new = exchange_optimization(
        paired, unpaired, iot_positions, iot_data_sizes,
        energy_dict['uav_energy']
    )

    if energy_new < energy_dict['uav_energy']:
        if verbose:
            print(f"  - Pairing improved! Energy reduced by {energy_dict['uav_energy'] - energy_new:.2f} J")
        paired = paired_new
        unpaired = unpaired_new

        # 重新计算能耗 (简化: 这里直接使用估计值)
        energy_dict['uav_energy'] = energy_new
    else:
        if verbose:
            print("  - No improvement from pairing exchange")
    
    # 返回结果
    result = {
        'paired': paired,
        'unpaired': unpaired,
        'powers': powers,
        'hover_positions': hover_positions,
        'trajectories': trajectories,
        'energy': energy_dict,
        'uav_energy': energy_dict['uav_energy'],      # UAV能耗 (优化目标)
        'iot_positions': iot_positions,
        'iot_data_sizes': iot_data_sizes
    }

    if verbose:
        print("\n" + "="*60)
        print(f"Algorithm 1 Completed!")
        print(f"UAV Energy (Optimization Objective): {result['uav_energy']:.2f} J")
        print("="*60)
    
    return result


# ============================================================================
# 10. 可视化
# ============================================================================

def visualize_results(result: Dict, save_path: str = None):
    """
    可视化优化结果
    
    Args:
        result: algorithm_1的返回结果
        save_path: 保存路径 (可选)
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 子图1: IoT设备和UAV位置
    ax1 = fig.add_subplot(131)
    
    iot_pos = result['iot_positions']
    hover_pos = result['hover_positions']
    
    # 绘制IoT设备
    paired_devices = set()
    for k, m in result['paired']:
        paired_devices.add(k)
        paired_devices.add(m)
    
    unpaired_devices = result['unpaired']
    
    # 配对设备 (蓝色)
    if len(paired_devices) > 0:
        paired_indices = list(paired_devices)
        ax1.scatter(iot_pos[paired_indices, 0], iot_pos[paired_indices, 1],
                   c='blue', marker='o', s=100, label='Paired IoT', alpha=0.6)
    
    # 未配对设备 (红色)
    if len(unpaired_devices) > 0:
        ax1.scatter(iot_pos[unpaired_devices, 0], iot_pos[unpaired_devices, 1],
                   c='red', marker='s', s=100, label='Unpaired IoT', alpha=0.6)
    
    # 绘制配对连线
    for k, m in result['paired']:
        ax1.plot([iot_pos[k, 0], iot_pos[m, 0]],
                [iot_pos[k, 1], iot_pos[m, 1]],
                'b--', alpha=0.3, linewidth=1)
    
    # 绘制UAV悬停点
    ax1.scatter(hover_pos[:, 0], hover_pos[:, 1],
               c='green', marker='^', s=200, label='UAV Hover', 
               edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('IoT Devices and UAV Positions', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 子图2: UAV轨迹
    ax2 = fig.add_subplot(132)
    
    colors = ['red', 'blue', 'green']
    start_pos = np.array([Config.AREA_SIZE/2, Config.AREA_SIZE/2, Config.H_U])
    
    for u, trajectory in enumerate(result['trajectories']):
        if len(trajectory) == 0:
            continue
        
        # 构建完整路径
        path_x = [start_pos[0]]
        path_y = [start_pos[1]]
        
        for point_idx in trajectory:
            path_x.append(hover_pos[point_idx, 0])
            path_y.append(hover_pos[point_idx, 1])
        
        # 返回起点
        path_x.append(start_pos[0])
        path_y.append(start_pos[1])
        
        ax2.plot(path_x, path_y, 'o-', color=colors[u % 3],
                linewidth=2, markersize=8, label=f'UAV {u+1}', alpha=0.7)
    
    # 起点
    ax2.scatter(start_pos[0], start_pos[1], c='black', marker='*',
               s=500, label='Start/End', edgecolors='yellow', linewidths=2)
    
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('UAV Trajectories', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 子图3: 能耗分布
    ax3 = fig.add_subplot(133)

    energy = result['energy']
    categories = ['UAV\nHovering', 'UAV\nFlight']
    values = [energy['hover_energy'], energy['flight_energy']]

    bars = ax3.bar(categories, values, color=['#4ECDC4', '#45B7D1'], alpha=0.8)

    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} J', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.set_ylabel('Energy (J)', fontsize=12)
    ax3.set_title(f'UAV Energy Breakdown\nTotal: {energy["uav_energy"]:.2f} J',
                 fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


# ============================================================================
# 11. 场景生成
# ============================================================================

def generate_scenario(K: int = 40, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成仿真场景
    
    Args:
        K: IoT设备数量
        seed: 随机种子
    
    Returns:
        iot_positions: [K, 3]
        iot_data_sizes: [K]
    """
    np.random.seed(seed)
    
    # IoT位置: 随机分布在500m × 500m区域
    iot_positions = np.random.uniform(0, Config.AREA_SIZE, (K, 2))
    iot_positions = np.column_stack([iot_positions, np.zeros(K)])
    
    # 数据包大小: 100KB - 1MB
    iot_data_sizes = np.random.uniform(Config.DATA_SIZE_MIN, 
                                       Config.DATA_SIZE_MAX, K)
    
    return iot_positions, iot_data_sizes


# ============================================================================
# 12. 主程序
# ============================================================================

def main():
    """主程序入口"""
    print("\n" + "="*70)
    print("IoT-UAV Phase Optimization Implementation")
    print("Paper: Joint UAV Trajectory Planning and LEO Satellite Selection")
    print("="*70 + "\n")
    
    # 生成场景
    K = 40  # IoT设备数量
    print(f"Generating scenario with {K} IoT devices...")
    iot_positions, iot_data_sizes = generate_scenario(K=K, seed=42)
    
    # 运行Algorithm 1
    result = algorithm_1(iot_positions, iot_data_sizes, verbose=True)
    
    # 可视化
    print("\nGenerating visualization...")
    visualize_results(result, save_path='algorithm1_result.png')
    
    # 输出统计信息
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total IoT devices: {K}")
    print(f"NOMA pairs: {len(result['paired'])}")
    print(f"OFDMA nodes: {len(result['unpaired'])}")
    print(f"Spectrum efficiency improvement: {len(result['paired'])*2 / K * 100:.1f}%")
    print(f"\nUAV Energy Consumption:")
    print(f"  - Hovering: {result['energy']['hover_energy']:.2f} J")
    print(f"  - Flight: {result['energy']['flight_energy']:.2f} J")
    print(f"  - TOTAL: {result['uav_energy']:.2f} J")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    result = main()