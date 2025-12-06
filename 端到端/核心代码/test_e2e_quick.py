"""
端到端实验快速测试脚本
测试单个场景，验证程序是否正常运行
"""

import sys
import os

# 添加路径
base_path = os.path.abspath('../..')
sys.path.insert(0, os.path.join(base_path, '算法1', '核心代码'))
sys.path.insert(0, os.path.join(base_path, '算法2', '核心代码'))

from end_to_end_experiment import run_single_experiment, calculate_e2e_metrics
from datetime import datetime, timezone
import numpy as np

def test_single_run():
    """测试单次实验运行"""

    print("\n" + "="*70)
    print("端到端实验快速测试")
    print("="*70)

    # 检查TLE文件
    tle_file = '../../starlink.tle'
    if not os.path.exists(tle_file):
        tle_file = 'starlink.tle'
        if not os.path.exists(tle_file):
            print("\n[警告] 未找到starlink.tle文件")
            print("请将starlink.tle文件放在以下位置之一:")
            print("  1. 项目根目录 (推荐)")
            print("  2. 端到端/核心代码 目录")
            print("\n下载地址:")
            print("  https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle")
            return False

    print(f"\n[✓] 找到TLE文件: {tle_file}")

    # 临时修改配置
    from end_to_end_experiment import E2EConfig
    E2EConfig.TLE_FILE = tle_file

    # 测试参数
    test_params = {
        'group_id': 'G1',
        'alg1_method': 'P1',
        'alg2_method': 'Demand-Aware',
        'K': 20,  # 使用较小的K加快测试
        'seed': 27,
        'start_time': datetime.now(timezone.utc)
    }

    print(f"\n测试配置:")
    print(f"  组别: {test_params['group_id']}")
    print(f"  算法1: {test_params['alg1_method']}")
    print(f"  算法2: {test_params['alg2_method']}")
    print(f"  IoT设备数: {test_params['K']}")
    print(f"  随机种子: {test_params['seed']}")

    try:
        # 运行测试
        result = run_single_experiment(**test_params)

        # 验证结果
        print("\n[✓] 实验运行成功!")
        print("\n关键结果:")
        print(f"  端到端时延: {result['e2e_metrics']['e2e_latency']:.2f} s")
        print(f"  系统能耗: {result['e2e_metrics']['system_energy']:.2f} J")
        print(f"  传输成功率: {result['e2e_metrics']['delivery_success_rate']*100:.3f} %")

        print("\n阶段详情:")
        print(f"  [算法1] 收集时间: {result['phase1']['collection_time']:.2f}s, "
              f"UAV能耗: {result['phase1']['uav_energy']:.2f}J")
        print(f"  [算法2] 切换次数: {result['phase2']['handover_count']}, "
              f"丢包率: {result['phase2']['packet_loss_rate']*100:.3f}%")

        return True

    except Exception as e:
        print(f"\n[✗] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_run()

    if success:
        print("\n" + "="*70)
        print("✓ 测试通过! 可以运行完整实验")
        print("\n运行完整实验:")
        print("  python end_to_end_experiment.py")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("✗ 测试失败，请检查配置")
        print("="*70 + "\n")
