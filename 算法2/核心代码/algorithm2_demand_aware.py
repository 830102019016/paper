"""
Algorithm 2 - Demand-Aware Handover Strategy
基于论文: Demand-Aware Flexible Handover Strategy for LEO Constellation

仿真场景:
- 对比两种卫星选择策略:
  1. Proposed (Greedy): 贪婪选择最大速率卫星
  2. Demand-Aware: 满足需求即可,最小化切换

核心思想:
- Greedy: 总是选择速率最高的卫星 → 频繁切换
- Demand-Aware: 只要当前卫星满足需求就保持连接 → 减少切换

作者: 复现实现
日期: 2025-11-26
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from scipy.interpolate import make_interp_spline

# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class SatelliteInfo:
    """卫星信息"""
    sat_id: int
    sat_name: str
    elevation_deg: float
    azimuth_deg: float
    distance_km: float
    data_rate_mbps: float = 0.0

@dataclass
class UAVLocation:
    """UAV位置"""
    lat: float  # 纬度
    lon: float  # 经度
    alt_m: float  # 高度(米)

@dataclass
class TransmissionRecord:
    """传输记录"""
    time_sec: float  # 时间(秒)
    data_rate_mbps: float  # 数据速率(Mbps)
    cumulative_data_mb: float  # 累计数据量(MB)
    satellite_id: int  # 当前使用的卫星ID
    distance_km: float  # 卫星距离

# ============================================================================
# 仿真参数
# ============================================================================

class SimulationParameters:
    """仿真参数 (论文Table I)"""

    # 地球和卫星参数
    R_EARTH_KM = 6378.0
    H_SAT_KM = 550.0
    H_UAV_KM = 0.2

    # 信道参数
    FREQ_GHZ = 20.0
    BANDWIDTH_HZ = 10e6  # 10 MHz
    PTR_W = 10.0
    GTR_DBI = 10.0
    GRE_DBI = 30.0
    NOISE_W = 4e-14

    # 约束参数
    THETA_MIN_DEG = 10.0  # 降低最小仰角约束，提高卫星可见性

# ============================================================================
# LEO卫星星座管理
# ============================================================================

class LEOConstellation:
    """LEO卫星星座管理"""

    def __init__(self, tle_file: str, num_sats: int = 200):
        try:
            from skyfield.api import load, EarthSatellite, wgs84
            self.wgs84 = wgs84
        except ImportError:
            raise ImportError("请安装skyfield: pip install skyfield")

        self.ts = load.timescale()
        self.num_sats = num_sats
        self.tle_file = tle_file
        self.satellites = self._load_satellites(tle_file)

        print(f"[OK] 成功加载 {len(self.satellites)} 颗LEO卫星")

    def _load_satellites(self, tle_file: str) -> List:
        """加载TLE文件并解析卫星"""
        from skyfield.api import EarthSatellite

        try:
            with open(tle_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"TLE文件未找到: {tle_file}")

        lines = [l for l in lines if l.strip()]
        satellites = []
        total_sats = len(lines) // 3

        import random
        random.seed(42)

        # 如果num_sats为-1或大于总数，加载所有卫星
        if self.num_sats == -1 or self.num_sats >= total_sats:
            indices = list(range(total_sats))
        else:
            indices = random.sample(range(total_sats), min(self.num_sats, total_sats))

        for idx in indices:
            i = idx * 3
            if i + 2 >= len(lines):
                break

            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()

            try:
                sat = EarthSatellite(line1, line2, name, self.ts)
                satellites.append({
                    'id': idx,
                    'name': name,
                    'satellite': sat
                })
            except Exception as e:
                continue

        return satellites

    def get_satellite_positions(self, uav_loc: UAVLocation,
                               time_utc: str) -> List[SatelliteInfo]:
        """计算所有卫星相对UAV的位置"""
        dt = datetime.strptime(time_utc, '%Y-%m-%d %H:%M:%S')
        t = self.ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

        uav_position = self.wgs84.latlon(
            uav_loc.lat,
            uav_loc.lon,
            elevation_m=uav_loc.alt_m
        )

        sat_infos = []

        for sat_data in self.satellites:
            sat = sat_data['satellite']

            try:
                difference = (sat - uav_position).at(t)
                topocentric = difference.altaz()

                elevation = topocentric[0].degrees
                azimuth = topocentric[1].degrees
                distance = topocentric[2].km

                sat_info = SatelliteInfo(
                    sat_id=sat_data['id'],
                    sat_name=sat_data['name'],
                    elevation_deg=elevation,
                    azimuth_deg=azimuth,
                    distance_km=distance
                )

                sat_infos.append(sat_info)

            except Exception as e:
                continue

        return sat_infos

# ============================================================================
# 信道模型
# ============================================================================

class ChannelModel:
    """UAV-LEO信道模型"""

    @staticmethod
    def calculate_received_power(distance_km: float) -> float:
        """计算接收功率 (Eq. 14)"""
        c = 3e8
        freq_hz = SimulationParameters.FREQ_GHZ * 1e9
        distance_m = distance_km * 1000

        wavelength = c / freq_hz
        free_space_loss = (wavelength / (4 * np.pi * distance_m)) ** 2

        gtr_linear = 10 ** (SimulationParameters.GTR_DBI / 10)
        gre_linear = 10 ** (SimulationParameters.GRE_DBI / 10)

        p_received = (SimulationParameters.PTR_W *
                     gtr_linear * gre_linear *
                     free_space_loss)

        return p_received

    @staticmethod
    def calculate_data_rate(distance_km: float) -> float:
        """计算数据速率 (Eq. 15, Shannon公式)"""
        p_re = ChannelModel.calculate_received_power(distance_km)
        snr = p_re / SimulationParameters.NOISE_W
        data_rate = SimulationParameters.BANDWIDTH_HZ * np.log2(1 + snr)
        return data_rate

# ============================================================================
# 卫星选择器基类
# ============================================================================

class SatelliteSelector:
    """卫星选择器基类"""

    def __init__(self, constellation: LEOConstellation,
                 theta_min: float = SimulationParameters.THETA_MIN_DEG):
        self.constellation = constellation
        self.theta_min = theta_min

    def filter_qualified_satellites(self,
                                    sat_infos: List[SatelliteInfo]) -> List[SatelliteInfo]:
        """筛选符合仰角约束的卫星"""
        qualified = []

        for sat in sat_infos:
            if sat.elevation_deg >= self.theta_min and sat.distance_km < 3000:
                sat.data_rate_mbps = ChannelModel.calculate_data_rate(
                    sat.distance_km
                ) / 1e6
                qualified.append(sat)

        return qualified

    def select(self, uav_loc: UAVLocation, time_utc: str) -> Optional[SatelliteInfo]:
        """选择卫星 (由子类实现)"""
        raise NotImplementedError

# ============================================================================
# 两种选择策略
# ============================================================================

class GreedySelector(SatelliteSelector):
    """Greedy方法: 贪婪选择吞吐量最大的卫星"""

    def select(self, uav_loc: UAVLocation, time_utc: str) -> Optional[SatelliteInfo]:
        """每次都选择当前吞吐量最大的卫星"""
        sat_infos = self.constellation.get_satellite_positions(uav_loc, time_utc)
        qualified = self.filter_qualified_satellites(sat_infos)

        if not qualified:
            return None

        # 贪婪选择: 吞吐量最大
        best_sat = max(qualified, key=lambda s: s.data_rate_mbps)
        return best_sat


class DemandAwareSelector(SatelliteSelector):
    """Demand-Aware方法: 满足需求即可,最小化切换"""

    def __init__(self, constellation: LEOConstellation,
                 demand_mbps: float = 200.0,
                 theta_min: float = SimulationParameters.THETA_MIN_DEG):
        """
        Args:
            constellation: LEO卫星星座
            demand_mbps: 用户需求速率 (Mbps)
            theta_min: 最小仰角约束 (度)
        """
        super().__init__(constellation, theta_min)
        self.demand_mbps = demand_mbps  # 用户需求速率
        self.current_sat_id = None  # 当前连接的卫星ID

    def select(self, uav_loc: UAVLocation, time_utc: str) -> Optional[SatelliteInfo]:
        """
        Demand-Aware选择逻辑:
        1. 如果当前卫星满足需求,保持连接 (不切换)
        2. 否则,从合格卫星中选择满足需求且速率最高的
        """
        sat_infos = self.constellation.get_satellite_positions(uav_loc, time_utc)
        qualified = self.filter_qualified_satellites(sat_infos)

        if not qualified:
            self.current_sat_id = None
            return None

        # 检查当前卫星是否仍可见且满足需求
        current_sat = None
        if self.current_sat_id is not None:
            for sat in qualified:
                if sat.sat_id == self.current_sat_id:
                    current_sat = sat
                    break

        # 条件检查: 当前卫星速率 >= 需求速率
        if current_sat is not None and current_sat.data_rate_mbps >= self.demand_mbps:
            # 满足需求,保持当前卫星 (不切换)
            return current_sat
        else:
            # 当前卫星无法满足需求,需要切换
            # 从合格卫星中选择满足需求的
            candidates = [s for s in qualified if s.data_rate_mbps >= self.demand_mbps]

            if not candidates:
                # 没有卫星能满足需求,选择速率最高的
                best_sat = max(qualified, key=lambda s: s.data_rate_mbps)
            else:
                # 从满足需求的卫星中选择速率最高的
                best_sat = max(candidates, key=lambda s: s.data_rate_mbps)

            self.current_sat_id = best_sat.sat_id
            return best_sat

# ============================================================================
# 仿真函数
# ============================================================================

def count_switches(records: List[TransmissionRecord]) -> int:
    """统计卫星切换次数"""
    switches = 0
    prev_sat_id = None
    for r in records:
        if prev_sat_id is not None and r.satellite_id != prev_sat_id and r.satellite_id != -1:
            switches += 1
        prev_sat_id = r.satellite_id
    return switches


def run_simulation(tle_file: str,
                   num_sats: int,
                   demand_mbps: float,
                   duration_minutes: int = 15,
                   time_step_sec: int = 30,
                   start_time: datetime = None) -> Tuple[Dict[str, List[TransmissionRecord]], Dict[str, int]]:
    """运行单次仿真,对比Greedy和Demand-Aware"""

    print(f"\n--- 运行仿真: {num_sats if num_sats != -1 else '所有'}颗卫星, 需求={demand_mbps}Mbps ---")

    # 创建两个独立的星座实例,避免共享状态
    constellation_greedy = LEOConstellation(tle_file, num_sats)
    constellation_demand = LEOConstellation(tle_file, num_sats)

    selectors = {
        'Greedy': GreedySelector(constellation_greedy),
        'Demand-Aware': DemandAwareSelector(constellation_demand, demand_mbps=demand_mbps)
    }

    # UAV固定位置
    uav_loc = UAVLocation(lat=15.0, lon=118.0, alt_m=200.0)

    # 使用固定起始时间
    if start_time is None:
        start_time = datetime.now(timezone.utc)

    num_steps = (duration_minutes * 60) // time_step_sec

    results = {'Greedy': [], 'Demand-Aware': []}
    cumulative_data = {'Greedy': 0.0, 'Demand-Aware': 0.0}

    for step in range(num_steps):
        current_time = start_time + timedelta(seconds=step * time_step_sec)
        time_utc = current_time.strftime('%Y-%m-%d %H:%M:%S')
        elapsed_sec = step * time_step_sec

        for method_name, selector in selectors.items():
            sat = selector.select(uav_loc, time_utc)

            if sat is not None:
                data_rate_mbps = sat.data_rate_mbps
                transmitted_mb = data_rate_mbps * time_step_sec / 8
                cumulative_data[method_name] += transmitted_mb

                record = TransmissionRecord(
                    time_sec=elapsed_sec,
                    data_rate_mbps=data_rate_mbps,
                    cumulative_data_mb=cumulative_data[method_name],
                    satellite_id=sat.sat_id,
                    distance_km=sat.distance_km
                )
            else:
                record = TransmissionRecord(
                    time_sec=elapsed_sec,
                    data_rate_mbps=0.0,
                    cumulative_data_mb=cumulative_data[method_name],
                    satellite_id=-1,
                    distance_km=0.0
                )
            results[method_name].append(record)

    # 统计切换次数
    switch_counts = {
        'Greedy': count_switches(results['Greedy']),
        'Demand-Aware': count_switches(results['Demand-Aware'])
    }

    # 输出统计
    for method_name in ['Greedy', 'Demand-Aware']:
        records = results[method_name]
        rates = [r.data_rate_mbps for r in records if r.data_rate_mbps > 0]
        if rates:
            print(f"  {method_name}: 累计={cumulative_data[method_name]:.1f}MB, "
                  f"平均速率={np.mean(rates):.2f}Mbps, 切换={switch_counts[method_name]}次")

    return results, switch_counts

# ============================================================================
# 绘图函数
# ============================================================================

def plot_comparison(all_results: Dict[str, Dict[str, List[TransmissionRecord]]],
                   all_switch_counts: Dict[str, Dict[str, int]],
                   scenario_name: str):
    """绘制对比图"""

    print(f"\n绘制{scenario_name}对比图...")

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    colors = {
        'Greedy': '#E74C3C',
        'Demand-Aware': '#2ECC71'
    }

    scenario_colors = {
        '200': '#E74C3C',
        '500': '#3498DB',
        '1500': '#2ECC71',
        'All': '#9B59B6'
    }

    if scenario_name == 'Number of Satellites':
        # 图1: 卫星数量对比 - 2行布局
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        scenarios = list(all_results.keys())

        # 获取需求速率 (从第一个场景的Demand-Aware选择器获取)
        demand_rate = 8.0  # 默认值，会从实际数据中获取

        # 第一行左侧: 各场景的Greedy速率曲线 (在同一个子图)
        ax1 = fig.add_subplot(gs[0, 0])
        for key in scenarios:
            results = all_results[key]
            records = results['Greedy']
            times = np.array([r.time_sec / 60 for r in records])
            rates = np.array([r.data_rate_mbps for r in records])

            if len(times) > 3:
                times_smooth = np.linspace(times.min(), times.max(), 300)
                spl = make_interp_spline(times, rates, k=3)
                rates_smooth = spl(times_smooth)
                ax1.plot(times_smooth, rates_smooth,
                        color=scenario_colors.get(key, '#000000'),
                        linewidth=2.5, alpha=0.8, label=f'{key} Sats')

            # 标记切换点 (标记切换前的最后一个点)
            switch_times = []
            switch_rates = []
            prev_sat_id = None
            prev_time = None
            prev_rate = None
            for r in records:
                if prev_sat_id is not None and r.satellite_id != prev_sat_id and r.satellite_id != -1:
                    # 标记切换前的时间和速率（旧卫星的最后一个点）
                    if prev_time is not None and prev_rate is not None:
                        switch_times.append(prev_time / 60)
                        switch_rates.append(prev_rate)
                prev_sat_id = r.satellite_id
                prev_time = r.time_sec
                prev_rate = r.data_rate_mbps

            if switch_times:
                ax1.scatter(switch_times, switch_rates, marker='v', s=80,
                           color=scenario_colors.get(key, '#000000'), edgecolors='white',
                           linewidths=1.5, zorder=5, alpha=0.8)

        # 添加需求速率参考线
        ax1.axhline(y=demand_rate, color='#FF6B6B', linestyle='--', linewidth=2,
                   label=f'Demand Rate ({demand_rate} Mbps)', alpha=0.7)

        ax1.set_title('Greedy - Data Rate', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time (min)', fontsize=11)
        ax1.set_ylabel('Rate (Mbps)', fontsize=11)
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 第一行右侧: 各场景的Demand-Aware速率曲线 (在同一个子图)
        ax2 = fig.add_subplot(gs[0, 1])
        for key in scenarios:
            results = all_results[key]
            records = results['Demand-Aware']
            times = np.array([r.time_sec / 60 for r in records])
            rates = np.array([r.data_rate_mbps for r in records])

            if len(times) > 3:
                times_smooth = np.linspace(times.min(), times.max(), 300)
                spl = make_interp_spline(times, rates, k=3)
                rates_smooth = spl(times_smooth)
                ax2.plot(times_smooth, rates_smooth,
                        color=scenario_colors.get(key, '#000000'),
                        linewidth=2.5, alpha=0.8, label=f'{key} Sats')

            # 标记切换点 (标记切换前的最后一个点，即速率谷底)
            switch_times = []
            switch_rates = []
            prev_sat_id = None
            prev_time = None
            prev_rate = None
            for r in records:
                if prev_sat_id is not None and r.satellite_id != prev_sat_id and r.satellite_id != -1:
                    # 标记切换前的时间和速率（旧卫星的最后一个点）
                    if prev_time is not None and prev_rate is not None:
                        switch_times.append(prev_time / 60)
                        switch_rates.append(prev_rate)
                prev_sat_id = r.satellite_id
                prev_time = r.time_sec
                prev_rate = r.data_rate_mbps

            if switch_times:
                ax2.scatter(switch_times, switch_rates, marker='s', s=80,
                           color=scenario_colors.get(key, '#000000'), edgecolors='white',
                           linewidths=1.5, zorder=5, alpha=0.8)

        # 添加需求速率参考线
        ax2.axhline(y=demand_rate, color='#FF6B6B', linestyle='--', linewidth=2,
                   label=f'Demand Rate ({demand_rate} Mbps)', alpha=0.7)

        ax2.set_title('Demand-Aware - Data Rate', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time (min)', fontsize=11)
        ax2.set_ylabel('Rate (Mbps)', fontsize=11)
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 第二行: 切换次数对比柱状图 (跨两列)
        ax3 = fig.add_subplot(gs[1, :])
        x = np.arange(len(scenarios))
        width = 0.35

        greedy_counts = [all_switch_counts[k]['Greedy'] for k in scenarios]
        demand_counts = [all_switch_counts[k]['Demand-Aware'] for k in scenarios]

        bars1 = ax3.bar(x - width/2, greedy_counts, width,
                       label='Greedy', color=colors['Greedy'], edgecolor='white')
        bars2 = ax3.bar(x + width/2, demand_counts, width,
                       label='Demand-Aware', color=colors['Demand-Aware'], edgecolor='white')

        for bar in bars1:
            height = bar.get_height()
            ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11)
        for bar in bars2:
            height = bar.get_height()
            ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11)

        ax3.set_xlabel('Number of Satellites', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Handover Count', fontsize=12, fontweight='bold')
        ax3.set_title('Handover Comparison', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

    else:
        # 图2: 不同需求速率对比 - 2行布局
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 定义需求速率对应的颜色
        demand_colors = {
            'Video Conf\n(4Mbps)': '#3498DB',      # 蓝色 - 视频会议
            '1080p Stream\n(8Mbps)': '#2ECC71',    # 绿色 - 1080p流
            '2K Stream\n(16Mbps)': '#F39C12',      # 橙色 - 2K流
            'Cloud Gaming\n(20Mbps)': '#9B59B6'    # 紫色 - 云游戏
        }

        scenarios = list(all_results.keys())

        # 提取各场景的需求速率值
        demand_rate_values = {
            'Video Conf\n(4Mbps)': 4.0,
            '1080p Stream\n(8Mbps)': 8.0,
            '2K Stream\n(16Mbps)': 16.0,
            'Cloud Gaming\n(20Mbps)': 20.0
        }

        # 第一行左侧: Greedy速率曲线 (多条线，不同需求场景)
        ax1 = fig.add_subplot(gs[0, 0])
        # 按需求速率从高到低排序，确保低速率曲线在上层
        sorted_scenarios = sorted(scenarios, key=lambda k: demand_rate_values.get(k, 0), reverse=True)

        for idx, key in enumerate(sorted_scenarios):
            results = all_results[key]
            records = results['Greedy']
            times = np.array([r.time_sec / 60 for r in records])
            rates = np.array([r.data_rate_mbps for r in records])

            if len(times) > 3:
                times_smooth = np.linspace(times.min(), times.max(), 300)
                spl = make_interp_spline(times, rates, k=3)
                rates_smooth = spl(times_smooth)
                ax1.plot(times_smooth, rates_smooth,
                        color=demand_colors.get(key, '#000000'),
                        linewidth=2.5, alpha=0.7, label=key, zorder=10-idx)

            # 标记切换点 (标记切换前的最后一个点)
            switch_times = []
            switch_rates = []
            prev_sat_id = None
            prev_time = None
            prev_rate = None
            for r in records:
                if prev_sat_id is not None and r.satellite_id != prev_sat_id and r.satellite_id != -1:
                    # 标记切换前的时间和速率（旧卫星的最后一个点）
                    if prev_time is not None and prev_rate is not None:
                        switch_times.append(prev_time / 60)
                        switch_rates.append(prev_rate)
                prev_sat_id = r.satellite_id
                prev_time = r.time_sec
                prev_rate = r.data_rate_mbps

            if switch_times:
                ax1.scatter(switch_times, switch_rates, marker='v', s=80,
                           color=demand_colors.get(key, '#000000'), edgecolors='white',
                           linewidths=1.5, zorder=15-idx, alpha=0.7)

            # 添加对应的需求速率参考线
            demand_val = demand_rate_values.get(key, 0)
            if demand_val > 0:
                ax1.axhline(y=demand_val, color=demand_colors.get(key, '#000000'),
                           linestyle=':', linewidth=1.5, alpha=0.4, zorder=1)

        ax1.set_title('Greedy - Data Rate (1500 Satellites)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time (min)', fontsize=11)
        ax1.set_ylabel('Rate (Mbps)', fontsize=11)
        ax1.legend(fontsize=9, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 第一行右侧: Demand-Aware速率曲线 (多条线，不同需求场景)
        ax2 = fig.add_subplot(gs[0, 1])
        # 按需求速率从高到低排序，确保低速率曲线在上层
        for idx, key in enumerate(sorted_scenarios):
            results = all_results[key]
            records = results['Demand-Aware']
            times = np.array([r.time_sec / 60 for r in records])
            rates = np.array([r.data_rate_mbps for r in records])

            if len(times) > 3:
                times_smooth = np.linspace(times.min(), times.max(), 300)
                spl = make_interp_spline(times, rates, k=3)
                rates_smooth = spl(times_smooth)
                ax2.plot(times_smooth, rates_smooth,
                        color=demand_colors.get(key, '#000000'),
                        linewidth=2.5, alpha=0.7, label=key, zorder=10-idx)

            # 标记切换点 (标记切换前的最后一个点，即速率谷底)
            switch_times = []
            switch_rates = []
            prev_sat_id = None
            prev_time = None
            prev_rate = None
            for r in records:
                if prev_sat_id is not None and r.satellite_id != prev_sat_id and r.satellite_id != -1:
                    # 标记切换前的时间和速率（旧卫星的最后一个点）
                    if prev_time is not None and prev_rate is not None:
                        switch_times.append(prev_time / 60)
                        switch_rates.append(prev_rate)
                prev_sat_id = r.satellite_id
                prev_time = r.time_sec
                prev_rate = r.data_rate_mbps

            if switch_times:
                ax2.scatter(switch_times, switch_rates, marker='s', s=80,
                           color=demand_colors.get(key, '#000000'), edgecolors='white',
                           linewidths=1.5, zorder=15-idx, alpha=0.7)

            # 添加对应的需求速率参考线
            demand_val = demand_rate_values.get(key, 0)
            if demand_val > 0:
                ax2.axhline(y=demand_val, color=demand_colors.get(key, '#000000'),
                           linestyle=':', linewidth=1.5, alpha=0.4, zorder=1)

        ax2.set_title('Demand-Aware - Data Rate (1500 Satellites)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time (min)', fontsize=11)
        ax2.set_ylabel('Rate (Mbps)', fontsize=11)
        ax2.legend(fontsize=9, loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 第二行: 切换次数对比柱状图 (跨两列，显示不同需求速率)
        ax3 = fig.add_subplot(gs[1, :])
        x = np.arange(len(scenarios))
        width = 0.35

        greedy_counts = [all_switch_counts[k]['Greedy'] for k in scenarios]
        demand_counts = [all_switch_counts[k]['Demand-Aware'] for k in scenarios]

        bars1 = ax3.bar(x - width/2, greedy_counts, width,
                       label='Greedy', color=colors['Greedy'], edgecolor='white')
        bars2 = ax3.bar(x + width/2, demand_counts, width,
                       label='Demand-Aware', color=colors['Demand-Aware'], edgecolor='white')

        for bar in bars1:
            height = bar.get_height()
            ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11)
        for bar in bars2:
            height = bar.get_height()
            ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11)

        ax3.set_xlabel('Demand Rate Scenario', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Handover Count', fontsize=12, fontweight='bold')
        ax3.set_title('Handover Comparison (1500 Satellites)', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios, fontsize=10)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 保存图表
    import os
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '结果图表')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'demand_aware_{scenario_name.lower().replace(" ", "_")}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] 图表已保存: {output_file}")

    plt.show()

# ============================================================================
# TLE下载工具
# ============================================================================

def download_starlink_tle(save_path: str = 'starlink.tle') -> bool:
    """下载Starlink TLE数据"""
    import requests

    print("\n下载Starlink TLE数据...")
    url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'

    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            with open(save_path, 'w') as f:
                f.write(response.text)

            lines = response.text.strip().split('\n')
            num_sats = len(lines) // 3

            print(f"[OK] 下载成功: {save_path}")
            print(f"  卫星数量: {num_sats}")
            return True
        else:
            print(f"[X] 下载失败: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"[X] 错误: {e}")
        return False

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""

    print("\n")
    print("+" + "="*58 + "+")
    print("|" + " "*8 + "Demand-Aware Handover 仿真对比" + " "*17 + "|")
    print("+" + "="*58 + "+")

    # Step 1: 准备TLE数据
    print("\n[Step 1] 准备TLE数据")
    tle_file = 'starlink.tle'

    import os
    if not os.path.exists(tle_file):
        print(f"未找到 {tle_file}, 开始下载...")
        if not download_starlink_tle(tle_file):
            print("\n下载失败, 请手动下载TLE文件")
            return
    else:
        print(f"[OK] TLE文件已存在: {tle_file}")

    # 使用固定起始时间保证可比性
    start_time = datetime.now(timezone.utc)

    # ========================================================================
    # 实验1: 不同卫星数量对比 (固定需求=8Mbps, 1080p视频流)
    # ========================================================================
    print("\n[Step 2] 实验1: 不同卫星数量对比")
    satellite_counts = [500, 1500, -1]  # 删除200颗卫星场景
    demand_mbps = 8.0  # 1080p视频流: 5-8 Mbps (Netflix建议)

    all_results_sat = {}
    all_switch_counts_sat = {}

    for num_sats in satellite_counts:
        try:
            results, switch_counts = run_simulation(
                tle_file,
                num_sats,
                demand_mbps=demand_mbps,
                duration_minutes=10,  # 缩短到10分钟，提高卫星可见率
                time_step_sec=20,     # 缩短采样间隔到20秒，提高连续性
                start_time=start_time
            )
            label = f'{num_sats}' if num_sats != -1 else 'All'
            all_results_sat[label] = results
            all_switch_counts_sat[label] = switch_counts
        except Exception as e:
            print(f"[X] 仿真失败 ({num_sats}颗卫星): {e}")
            import traceback
            traceback.print_exc()
            continue

    # 绘制卫星数量对比图
    print("\n[Step 3] 绘制卫星数量对比图")
    try:
        plot_comparison(all_results_sat, all_switch_counts_sat, 'Number of Satellites')
    except Exception as e:
        print(f"[X] 绘图失败: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # 实验2: 不同业务场景需求对比 (固定卫星数量=1500)
    # ========================================================================
    print("\n[Step 4] 实验2: 不同业务场景需求对比")
    demand_levels = [
        8.0,   # 视频流 1080p (Netflix: 5-8 Mbps)
        16.0,  # 视频流 2K/1440p (YouTube: 16-20 Mbps)
        20.0   # 云游戏 1080p/60fps (Stadia: 10-20 Mbps)
    ]
    num_sats = 1500  # 使用1500颗卫星，不同需求会选择不同卫星，曲线有差异

    all_results_demand = {}
    all_switch_counts_demand = {}

    # 业务场景标签
    demand_labels = {
        8.0: '1080p Stream\n(8Mbps)',
        16.0: '2K Stream\n(16Mbps)',
        20.0: 'Cloud Gaming\n(20Mbps)'
    }

    for demand in demand_levels:
        try:
            results, switch_counts = run_simulation(
                tle_file,
                num_sats,
                demand_mbps=demand,
                duration_minutes=10,  # 缩短到10分钟，提高卫星可见率
                time_step_sec=20,     # 缩短采样间隔到20秒，提高连续性
                start_time=start_time
            )
            label = demand_labels.get(demand, f'{demand}Mbps')
            all_results_demand[label] = results
            all_switch_counts_demand[label] = switch_counts
        except Exception as e:
            print(f"[X] 仿真失败 (需求={demand}Mbps): {e}")
            import traceback
            traceback.print_exc()
            continue

    # 绘制需求速率对比图
    print("\n[Step 5] 绘制需求速率对比图")
    try:
        plot_comparison(all_results_demand, all_switch_counts_demand, 'Demand Rate')
    except Exception as e:
        print(f"[X] 绘图失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("仿真完成!")
    print("="*60)


if __name__ == "__main__":
    main()
