"""
灰狼优化算法 (Grey Wolf Optimizer) for TSP
基于论文: Mirjalili et al. "Grey Wolf Optimizer" (2014)

用于优化UAV访问多个悬停点的轨迹规划（TSP问题）
"""

import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class GreyWolfOptimizer:
    """
    灰狼优化算法（GWO）

    算法原理：
    1. 狼群分为4个等级：Alpha(α), Beta(β), Delta(δ), Omega(ω)
    2. Alpha是最优解，Beta和Delta是次优解
    3. 其他狼（Omega）跟随Alpha、Beta、Delta移动
    4. 参数a从2线性递减到0，控制探索和开发
    """

    def __init__(self, n_wolves: int = 30, max_iter: int = 100, seed: int = None):
        """
        Args:
            n_wolves: 狼群数量（种群大小）
            max_iter: 最大迭代次数
            seed: 随机种子（可重复实验）
        """
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

    def optimize_tsp(self, distance_matrix: np.ndarray, verbose: bool = False) -> Tuple[List[int], float]:
        """
        使用GWO求解TSP问题

        Args:
            distance_matrix: 距离矩阵 [N, N]
            verbose: 是否打印迭代信息

        Returns:
            best_route: 最优路径 [0, 3, 1, 4, 2, ...]
            best_distance: 最优路径长度
        """
        n_cities = len(distance_matrix)

        # 初始化狼群（随机路径）
        wolves = [np.random.permutation(n_cities) for _ in range(self.n_wolves)]

        # 评估初始适应度
        fitness = np.array([self._evaluate_route(route, distance_matrix) for route in wolves])

        # 找出Alpha, Beta, Delta狼（前三名）
        sorted_indices = np.argsort(fitness)
        alpha_idx = sorted_indices[0]
        beta_idx = sorted_indices[1]
        delta_idx = sorted_indices[2]

        alpha = wolves[alpha_idx].copy()
        beta = wolves[beta_idx].copy()
        delta = wolves[delta_idx].copy()
        alpha_fitness = fitness[alpha_idx]
        beta_fitness = fitness[beta_idx]
        delta_fitness = fitness[delta_idx]

        # 记录最优历史
        best_fitness_history = [alpha_fitness]

        # 迭代优化
        for iteration in range(self.max_iter):
            # 线性递减的a参数（从2到0）
            a = 2 - iteration * (2 / self.max_iter)

            for i in range(self.n_wolves):
                # 更新每只狼的位置（基于Alpha, Beta, Delta）
                wolves[i] = self._update_position(
                    wolves[i], alpha, beta, delta, a, distance_matrix
                )

                # 评估新适应度
                new_fitness = self._evaluate_route(wolves[i], distance_matrix)
                fitness[i] = new_fitness

                # 更新Alpha, Beta, Delta
                if new_fitness < alpha_fitness:
                    # 新的最优解
                    delta = beta.copy()
                    delta_fitness = beta_fitness
                    beta = alpha.copy()
                    beta_fitness = alpha_fitness
                    alpha = wolves[i].copy()
                    alpha_fitness = new_fitness
                elif new_fitness < beta_fitness:
                    # 新的次优解
                    delta = beta.copy()
                    delta_fitness = beta_fitness
                    beta = wolves[i].copy()
                    beta_fitness = new_fitness
                elif new_fitness < delta_fitness:
                    # 新的第三优解
                    delta = wolves[i].copy()
                    delta_fitness = new_fitness

            best_fitness_history.append(alpha_fitness)

            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}/{self.max_iter}: Best Distance = {alpha_fitness:.2f}m")

        if verbose:
            print(f"  GWO Converged: Final Best Distance = {alpha_fitness:.2f}m")

        return alpha.tolist(), alpha_fitness

    def _evaluate_route(self, route: np.ndarray, dist_matrix: np.ndarray) -> float:
        """
        计算路径总长度（适应度函数）

        Args:
            route: 路径（城市访问顺序）
            dist_matrix: 距离矩阵

        Returns:
            总距离
        """
        total_dist = 0
        for i in range(len(route)):
            j = (i + 1) % len(route)  # 回到起点
            total_dist += dist_matrix[route[i], route[j]]
        return total_dist

    def _update_position(self, wolf: np.ndarray, alpha: np.ndarray,
                        beta: np.ndarray, delta: np.ndarray,
                        a: float, dist_matrix: np.ndarray) -> np.ndarray:
        """
        更新狼的位置（路径）

        使用交换序列方法（Swap Sequence）适配TSP离散问题

        Args:
            wolf: 当前狼的路径
            alpha, beta, delta: 领导狼的路径
            a: 控制参数（2→0）
            dist_matrix: 距离矩阵

        Returns:
            更新后的路径
        """
        # 计算向Alpha, Beta, Delta移动的交换序列
        swap_seq_alpha = self._get_swap_sequence(wolf, alpha)
        swap_seq_beta = self._get_swap_sequence(wolf, beta)
        swap_seq_delta = self._get_swap_sequence(wolf, delta)

        # GWO的位置更新公式（连续域）：
        # X(t+1) = (X1 + X2 + X3) / 3
        # 其中 Xi = X_leader - A * |C * X_leader - X|
        #
        # 离散域改造：通过控制应用交换序列的比例来模拟

        # 随机系数
        r1, r2 = np.random.random(), np.random.random()
        A = 2 * a * r1 - a  # A ∈ [-a, a]
        C = 2 * r2          # C ∈ [0, 2]

        # 计算应用交换的比例（基于A和C）
        # |A| < 1: exploitation（开发，局部搜索）
        # |A| >= 1: exploration（探索，全局搜索）
        ratio = abs(C - abs(A)) / 2.0  # 归一化到[0, 1]

        # 从当前wolf开始，向三个领导狼方向移动
        new_wolf = wolf.copy()

        # 向Alpha移动
        n_swaps_alpha = int(len(swap_seq_alpha) * ratio * 0.5)
        for swap in swap_seq_alpha[:n_swaps_alpha]:
            i, j = swap
            if i < len(new_wolf) and j < len(new_wolf):
                new_wolf[i], new_wolf[j] = new_wolf[j], new_wolf[i]

        # 向Beta移动
        n_swaps_beta = int(len(swap_seq_beta) * ratio * 0.3)
        for swap in swap_seq_beta[:n_swaps_beta]:
            i, j = swap
            if i < len(new_wolf) and j < len(new_wolf):
                new_wolf[i], new_wolf[j] = new_wolf[j], new_wolf[i]

        # 向Delta移动
        n_swaps_delta = int(len(swap_seq_delta) * ratio * 0.2)
        for swap in swap_seq_delta[:n_swaps_delta]:
            i, j = swap
            if i < len(new_wolf) and j < len(new_wolf):
                new_wolf[i], new_wolf[j] = new_wolf[j], new_wolf[i]

        # 局部搜索（2-opt优化）：概率性应用
        if abs(A) < 1 and np.random.random() < 0.2:
            new_wolf = self._two_opt_move(new_wolf, dist_matrix)
        elif abs(A) >= 1 and np.random.random() < 0.1:
            # 探索阶段：随机扰动
            i, j = np.random.choice(len(new_wolf), 2, replace=False)
            new_wolf[i], new_wolf[j] = new_wolf[j], new_wolf[i]

        return new_wolf

    def _get_swap_sequence(self, route1: np.ndarray, route2: np.ndarray) -> List[Tuple[int, int]]:
        """
        计算将route1变换为route2所需的交换序列

        使用贪心方法逐位置匹配

        Args:
            route1: 起始路径
            route2: 目标路径

        Returns:
            交换序列 [(i1, j1), (i2, j2), ...]
        """
        swaps = []
        current = route1.copy()

        for i in range(len(route2)):
            if current[i] != route2[i]:
                # 找到route2[i]在current中的位置
                j = np.where(current == route2[i])[0]
                if len(j) > 0:
                    j = j[0]
                    # 交换
                    current[i], current[j] = current[j], current[i]
                    swaps.append((i, j))

        return swaps

    def _two_opt_move(self, route: np.ndarray, dist_matrix: np.ndarray) -> np.ndarray:
        """
        2-opt局部搜索：随机选择两个边，尝试反转中间部分

        Args:
            route: 当前路径
            dist_matrix: 距离矩阵

        Returns:
            改进后的路径
        """
        n = len(route)
        if n <= 3:
            return route

        # 随机选择两个位置
        i, j = sorted(np.random.choice(n, 2, replace=False))

        if i == j or i + 1 == j:
            return route

        # 计算原距离
        old_dist = (dist_matrix[route[i], route[i + 1]] +
                   dist_matrix[route[j], route[(j + 1) % n]])

        # 计算反转后的距离
        new_dist = (dist_matrix[route[i], route[j]] +
                   dist_matrix[route[i + 1], route[(j + 1) % n]])

        # 如果有改进，执行反转
        if new_dist < old_dist:
            route[i + 1:j + 1] = route[i + 1:j + 1][::-1]

        return route


# ============================================================================
# TSP求解器（支持多UAV）
# ============================================================================

def gwo_tsp(hover_points: np.ndarray, num_uavs: int = 1,
           start_pos: np.ndarray = None, n_wolves: int = 30,
           max_iter: int = 100, verbose: bool = False) -> List[List[int]]:
    """
    使用GWO求解多UAV的TSP问题

    Args:
        hover_points: 悬停点坐标 [N, 3]
        num_uavs: UAV数量
        start_pos: 起始位置 [3]
        n_wolves: 狼群数量
        max_iter: 最大迭代次数
        verbose: 是否打印详细信息

    Returns:
        trajectories: 每个UAV的访问顺序列表
                     [[点0, 点3, 点7], [点1, 点4], ...]
    """
    N = len(hover_points)

    if N == 0:
        return [[] for _ in range(num_uavs)]

    if start_pos is None:
        start_pos = np.array([250, 250, 200])

    if verbose:
        print(f"\n[GWO-TSP] 悬停点: {N}, UAVs: {num_uavs}, 狼群: {n_wolves}, 迭代: {max_iter}")

    # 计算距离矩阵
    all_points = np.vstack([start_pos.reshape(1, -1), hover_points])
    dist_matrix_full = np.linalg.norm(
        all_points[:, np.newaxis, :] - all_points[np.newaxis, :, :],
        axis=2
    )

    if num_uavs == 1:
        # 单UAV情况：直接求解TSP
        if verbose:
            print("  模式: 单UAV-TSP")

        dist_matrix = dist_matrix_full[1:, 1:]  # 去除起点

        gwo = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter)
        route, distance = gwo.optimize_tsp(dist_matrix, verbose=verbose)

        return [route]

    else:
        # 多UAV: 使用K-means聚类 + 独立GWO
        if verbose:
            print("  模式: 多UAV-K-means+GWO")

        # 聚类分配悬停点
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_uavs, random_state=42, n_init=10)
            labels = kmeans.fit_predict(hover_points)
        except ImportError:
            # 如果没有sklearn，使用简单的顺序分配
            if verbose:
                print("  警告: sklearn未安装，使用顺序分配")
            labels = np.arange(N) % num_uavs

        trajectories = []
        gwo = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter)

        for u in range(num_uavs):
            cluster_indices = np.where(labels == u)[0]

            if len(cluster_indices) == 0:
                trajectories.append([])
                if verbose:
                    print(f"  UAV {u + 1}: 无分配点")
                continue

            if len(cluster_indices) == 1:
                # 只有一个点，直接访问（确保是int类型）
                trajectories.append([int(cluster_indices[0])])
                if verbose:
                    print(f"  UAV {u + 1}: 1个点 (直接访问)")
                continue

            # 提取子距离矩阵（包括起点）
            sub_indices = np.concatenate([[0], cluster_indices + 1])
            sub_dist_matrix_full = dist_matrix_full[np.ix_(sub_indices, sub_indices)]
            sub_dist_matrix = sub_dist_matrix_full[1:, 1:]  # 去除起点

            # GWO求解
            if verbose:
                print(f"  UAV {u + 1}: {len(cluster_indices)}个点")

            local_route, local_distance = gwo.optimize_tsp(sub_dist_matrix, verbose=False)

            # 映射回全局索引（确保是int类型）
            global_route = [int(cluster_indices[i]) for i in local_route]
            trajectories.append(global_route)

            if verbose:
                print(f"    → 路径长度: {local_distance:.2f}m")

        return trajectories


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GWO-TSP 测试")
    print("="*70)

    # 生成测试数据
    np.random.seed(42)
    n_points = 15
    hover_points = np.random.uniform(0, 500, (n_points, 3))
    hover_points[:, 2] = 200  # 固定高度

    # 测试1: 单UAV
    print("\n测试1: 单UAV-GWO")
    print("-" * 70)
    trajectories_gwo = gwo_tsp(
        hover_points,
        num_uavs=1,
        n_wolves=20,
        max_iter=50,
        verbose=True
    )
    print(f"GWO轨迹: {trajectories_gwo[0]}")

    # 测试2: 多UAV
    print("\n测试2: 多UAV-GWO")
    print("-" * 70)
    trajectories_multi = gwo_tsp(
        hover_points,
        num_uavs=3,
        n_wolves=20,
        max_iter=50,
        verbose=True
    )
    for u, traj in enumerate(trajectories_multi):
        print(f"UAV {u + 1} 轨迹: {traj}")

    # 对比最近邻方法（如果可用）
    try:
        from algorithm1 import nearest_neighbor_tsp, compute_trajectory_length

        print("\n对比: GWO vs 最近邻")
        print("-" * 70)

        # 最近邻
        start_pos = np.array([250, 250, 200])
        traj_nn = nearest_neighbor_tsp(hover_points, num_uavs=1, start_pos=start_pos)
        dist_nn = compute_trajectory_length(traj_nn[0], hover_points, start_pos)

        # GWO
        traj_gwo = gwo_tsp(hover_points, num_uavs=1, n_wolves=20, max_iter=50)
        dist_gwo = compute_trajectory_length(traj_gwo[0], hover_points, start_pos)

        print(f"最近邻距离: {dist_nn:.2f}m")
        print(f"GWO距离:    {dist_gwo:.2f}m")
        print(f"改进:       {(dist_nn - dist_gwo) / dist_nn * 100:.2f}%")

    except ImportError:
        print("\n(algorithm1.py 未找到，跳过对比)")

    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)
