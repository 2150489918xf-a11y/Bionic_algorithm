# -*- coding: utf-8 -*-
"""
蚁群算法求解旅行商问题 (ACO for TSP)
简单版 - 单文件实现
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ============================================================
#                    参数调试区 (Parameter Settings)
# ============================================================

# --- 数据文件设置 ---
# 可选: berlin52.tsp, eil51.tsp, st70.tsp, kroA100.tsp, att48.tsp
TSP_FILE = "data/berlin52.tsp"  # TSP数据文件路径

# --- 蚁群算法参数 ---
NUM_ANTS = 30           # 蚂蚁数量
NUM_ITERATIONS = 200    # 迭代次数
ALPHA = 1.0             # 信息素重要程度因子
BETA = 5.0              # 启发函数重要程度因子
RHO = 0.5               # 信息素挥发系数 (0-1)
Q = 100                 # 信息素常数

# --- 输出设置 ---
OUTPUT_DIR = "output"   # 输出目录

# --- 可视化设置 ---
FIGURE_DPI = 150        # 图片分辨率
SHOW_PLOTS = True       # 是否显示图片窗口

# ============================================================
#                    数据读取模块
# ============================================================

def read_tsp_file(filepath):
    """
    读取TSPLIB格式的TSP文件
    返回: (cities, known_optimal, optimal_tour)
    - cities: 城市坐标数组
    - known_optimal: 已知最优解距离
    - optimal_tour: 已知最优路径 (0-indexed)
    """
    cities = []
    known_optimal = None
    optimal_tour = None
    reading_coords = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # 读取已知最优解
            if line.startswith("KNOWN_OPTIMAL:"):
                known_optimal = float(line.split(":")[1].strip())
                continue

            # 读取最优路径 (1-indexed -> 0-indexed)
            if line.startswith("OPTIMAL_TOUR:"):
                tour_str = line.split(":")[1].strip()
                optimal_tour = [int(x) - 1 for x in tour_str.split()]
                continue

            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            if line == "EOF":
                break
            if reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    cities.append((x, y))

    return np.array(cities), known_optimal, optimal_tour


def calculate_distance_matrix(cities):
    """计算城市间距离矩阵"""
    n = len(cities)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = np.sqrt(
                    (cities[i][0] - cities[j][0])**2 +
                    (cities[i][1] - cities[j][1])**2
                )

    return dist_matrix


# ============================================================
#                    蚁群算法核心
# ============================================================

class AntColonyTSP:
    def __init__(self, dist_matrix, num_ants, num_iterations,
                 alpha, beta, rho, q):
        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

        # 初始化信息素矩阵
        self.pheromone = np.ones((self.num_cities, self.num_cities))

        # 启发函数 (距离的倒数)
        self.heuristic = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.heuristic[i][j] = 1.0 / dist_matrix[i][j]

        # 记录最优解
        self.best_route = None
        self.best_distance = float('inf')
        self.convergence_history = []

    def select_next_city(self, current_city, visited):
        """根据概率选择下一个城市"""
        probabilities = []
        unvisited = [c for c in range(self.num_cities) if c not in visited]

        for city in unvisited:
            tau = self.pheromone[current_city][city] ** self.alpha
            eta = self.heuristic[current_city][city] ** self.beta
            probabilities.append(tau * eta)

        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()

        return np.random.choice(unvisited, p=probabilities)

    def construct_route(self):
        """构建一条完整路径"""
        start_city = np.random.randint(0, self.num_cities)
        route = [start_city]
        visited = set(route)

        while len(route) < self.num_cities:
            next_city = self.select_next_city(route[-1], visited)
            route.append(next_city)
            visited.add(next_city)

        return route

    def calculate_route_distance(self, route):
        """计算路径总距离"""
        distance = 0
        for i in range(len(route)):
            distance += self.dist_matrix[route[i]][route[(i+1) % len(route)]]
        return distance

    def update_pheromone(self, all_routes, all_distances):
        """更新信息素"""
        # 信息素挥发
        self.pheromone *= (1 - self.rho)

        # 信息素增加
        for route, distance in zip(all_routes, all_distances):
            delta = self.q / distance
            for i in range(len(route)):
                city_a = route[i]
                city_b = route[(i+1) % len(route)]
                self.pheromone[city_a][city_b] += delta
                self.pheromone[city_b][city_a] += delta

    def run(self):
        """运行蚁群算法"""
        print("开始运行蚁群算法...")
        print(f"城市数量: {self.num_cities}")
        print(f"蚂蚁数量: {self.num_ants}")
        print(f"迭代次数: {self.num_iterations}")
        print("-" * 40)

        for iteration in range(self.num_iterations):
            all_routes = []
            all_distances = []

            # 每只蚂蚁构建路径
            for _ in range(self.num_ants):
                route = self.construct_route()
                distance = self.calculate_route_distance(route)
                all_routes.append(route)
                all_distances.append(distance)

                # 更新最优解
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_route = route.copy()

            # 更新信息素
            self.update_pheromone(all_routes, all_distances)

            # 记录收敛历史
            self.convergence_history.append(self.best_distance)

            # 打印进度
            if (iteration + 1) % 20 == 0 or iteration == 0:
                print(f"迭代 {iteration+1:4d}/{self.num_iterations}: "
                      f"最优距离 = {self.best_distance:.2f}")

        print("-" * 40)
        print(f"算法结束! 最优距离: {self.best_distance:.2f}")

        return self.best_route, self.best_distance


# ============================================================
#                    可视化模块
# ============================================================

def get_run_number(output_dir):
    """获取当前运行次数"""
    if not os.path.exists(output_dir):
        return 1
    existing = [d for d in os.listdir(output_dir) if d.startswith("第") and d.endswith("次运行")]
    if not existing:
        return 1
    numbers = []
    for d in existing:
        try:
            num = int(d.replace("第", "").replace("次运行", ""))
            numbers.append(num)
        except ValueError:
            pass
    return max(numbers) + 1 if numbers else 1


def plot_convergence_with_params(history, runtime, num_cities, known_optimal, output_path):
    """绘制收敛曲线图（包含参数信息）"""
    # 创建带有子图的figure，上面是图，下面是参数
    fig = plt.figure(figsize=(10, 9))

    # 上方：收敛曲线图
    ax = fig.add_axes([0.1, 0.35, 0.85, 0.58])  # [left, bottom, width, height]

    # 绘制收敛曲线
    ax.plot(range(1, len(history)+1), history, 'b-', linewidth=1.5, label='Best Distance')
    # 绘制已知最优解的红色水平线
    if known_optimal:
        ax.axhline(y=known_optimal, color='r', linestyle='--', linewidth=2, label=f'Known Optimal ({known_optimal})')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Distance', fontsize=12)
    ax.set_title('ACO Convergence Curve', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 下方：参数信息区域（无背景色，水平排列）
    param_ax = fig.add_axes([0.1, 0.02, 0.85, 0.28])  # 与x轴对齐
    param_ax.axis('off')

    # 参数文本（分两列显示）
    left_text = (
        f"Parameters:\n"
        f"  Ants: {NUM_ANTS}\n"
        f"  Iterations: {NUM_ITERATIONS}\n"
        f"  Alpha: {ALPHA}\n"
        f"  Beta: {BETA}\n"
        f"  Rho: {RHO}\n"
        f"  Q: {Q}"
    )

    if known_optimal:
        gap = ((history[-1] - known_optimal) / known_optimal * 100)
        right_text = (
            f"Results:\n"
            f"  Cities: {num_cities}\n"
            f"  Best Distance: {history[-1]:.2f}\n"
            f"  Known Optimal: {known_optimal}\n"
            f"  Gap: {gap:.2f}%\n"
            f"  Runtime: {runtime:.2f}s"
        )
    else:
        right_text = (
            f"Results:\n"
            f"  Cities: {num_cities}\n"
            f"  Best Distance: {history[-1]:.2f}\n"
            f"  Runtime: {runtime:.2f}s"
        )

    # 左侧参数
    param_ax.text(0.15, 0.9, left_text, transform=param_ax.transAxes, fontsize=11,
                  verticalalignment='top', horizontalalignment='left', family='monospace')
    # 右侧结果
    param_ax.text(0.55, 0.9, right_text, transform=param_ax.transAxes, fontsize=11,
                  verticalalignment='top', horizontalalignment='left', family='monospace')

    # 添加分隔线
    param_ax.axhline(y=0.95, xmin=0.02, xmax=0.98, color='gray', linewidth=1)

    plt.savefig(output_path, dpi=FIGURE_DPI)
    print(f"收敛曲线图已保存: {output_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_routes_comparison(cities, current_route, current_distance, optimal_route, optimal_distance, output_path):
    """绘制路径对比图：左边本次运行路径，右边已知最优路径"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 左图：本次运行路径 ---
    ax1.scatter(cities[:, 0], cities[:, 1], c='red', s=50, zorder=5)
    for i, (x, y) in enumerate(cities):
        ax1.annotate(str(i+1), (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    route_cities = cities[current_route + [current_route[0]]]
    ax1.plot(route_cities[:, 0], route_cities[:, 1], 'b-', linewidth=1.5, alpha=0.7)
    start = cities[current_route[0]]
    ax1.scatter([start[0]], [start[1]], c='green', s=150, marker='*', zorder=6, label='Start')
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.set_title(f'Current Run Route (Distance: {current_distance:.2f})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 右图：已知最优路径 ---
    ax2.scatter(cities[:, 0], cities[:, 1], c='red', s=50, zorder=5)
    for i, (x, y) in enumerate(cities):
        ax2.annotate(str(i+1), (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    if optimal_route:
        opt_route_cities = cities[optimal_route + [optimal_route[0]]]
        ax2.plot(opt_route_cities[:, 0], opt_route_cities[:, 1], 'g-', linewidth=1.5, alpha=0.7)
        opt_start = cities[optimal_route[0]]
        ax2.scatter([opt_start[0]], [opt_start[1]], c='green', s=150, marker='*', zorder=6, label='Start')
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)
    ax2.set_title(f'Known Optimal Route (Distance: {optimal_distance})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI)
    print(f"路径对比图已保存: {output_path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ============================================================
#                    主程序
# ============================================================

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(script_dir, OUTPUT_DIR)

    # 确保输出目录存在
    os.makedirs(output_base, exist_ok=True)

    # 获取运行次数并创建本次运行的输出目录
    run_number = get_run_number(output_base)
    run_dir = os.path.join(output_base, f"第{run_number}次运行")
    os.makedirs(run_dir, exist_ok=True)
    print(f"本次运行输出目录: {run_dir}")
    print("")

    # 读取TSP数据（包含最优解信息）
    tsp_path = os.path.join(script_dir, TSP_FILE)
    print(f"读取数据文件: {tsp_path}")
    cities, known_optimal, optimal_tour = read_tsp_file(tsp_path)
    print(f"成功读取 {len(cities)} 个城市")
    if known_optimal:
        print(f"已知最优解: {known_optimal}")
    print("")

    # 计算距离矩阵
    dist_matrix = calculate_distance_matrix(cities)

    # 创建蚁群算法实例
    aco = AntColonyTSP(
        dist_matrix=dist_matrix,
        num_ants=NUM_ANTS,
        num_iterations=NUM_ITERATIONS,
        alpha=ALPHA,
        beta=BETA,
        rho=RHO,
        q=Q
    )

    # 运行算法并计时
    start_time = time.time()
    best_route, best_distance = aco.run()
    runtime = time.time() - start_time

    # 生成输出文件路径
    convergence_path = os.path.join(run_dir, "convergence.png")
    routes_path = os.path.join(run_dir, "routes_comparison.png")

    # 绘制收敛曲线图（包含参数信息）
    plot_convergence_with_params(aco.convergence_history, runtime, len(cities), known_optimal, convergence_path)

    # 绘制路径对比图（本次运行 vs 已知最优）
    plot_routes_comparison(cities, best_route, best_distance,
                          optimal_tour, known_optimal, routes_path)

    print("")
    print(f"第 {run_number} 次运行完成!")
    print(f"输出目录: {run_dir}")


if __name__ == "__main__":
    main()
