"""
端到端实验启动脚本
自动配置路径并运行完整实验
"""

import sys
import os

# 设置工作目录为项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(project_root)

# 添加算法模块路径
sys.path.insert(0, os.path.join(project_root, '算法1', '核心代码'))
sys.path.insert(0, os.path.join(project_root, '算法2', '核心代码'))

print(f"[信息] 工作目录: {os.getcwd()}")
print(f"[信息] 项目根目录: {project_root}")

# 导入主程序
from end_to_end_experiment import main, E2EConfig

# 配置TLE文件路径（使用项目根目录下的文件）
E2EConfig.TLE_FILE = os.path.join(project_root, 'starlink.tle')

# 检查TLE文件
if not os.path.exists(E2EConfig.TLE_FILE):
    print(f"\n[错误] 未找到TLE文件: {E2EConfig.TLE_FILE}")
    print("\n请下载Starlink TLE文件:")
    print("  方法1: 访问 https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle")
    print("  方法2: 运行以下Python代码:")
    print("    from algorithm2_demand_aware import download_starlink_tle")
    print("    download_starlink_tle('starlink.tle')")
    sys.exit(1)

print(f"[信息] TLE文件: {E2EConfig.TLE_FILE}")
print(f"[信息] 实验配置: K={E2EConfig.K_VALUES}, Seeds={E2EConfig.RANDOM_SEEDS}")
print()

# 运行实验
if __name__ == "__main__":
    all_results, statistics = main()
