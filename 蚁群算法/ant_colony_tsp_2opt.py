# -*- coding: utf-8 -*-
"""
蚁群算法 + 2-opt局部搜索 求解旅行商问题 (ACO + 2-opt for TSP)
混合算法版本 - 单文件实现

2-opt算法说明:
2-opt是一种局部搜索优化算法，通过反转路径中的一段来消除交叉边，
从而改善路径质量。结合ACO可以显著提高解的质量。
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

# --- 2-opt参数 ---
USE_2OPT = True                 # 是否启用2-opt优化
OPT_2_START_ITER = 150          # 从第几代开始启用2-opt (让ACO先收敛一段时间)
OPT_2_FREQUENCY = 5             # 每隔多少代对最优解进行2-opt优化
OPT_2_FOR_ALL_ANTS = False      # 是否对每只蚂蚁的路径都进行2-opt (True会更慢但效果更好)

# --- 输出设置 ---
OUTPUT_DIR = "output_2opt"      # 输出目录

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
#                    2-opt局部搜索算法
# ============================================================

def two_opt_swap(route, i, k):
    """
    执行2-opt交换: 反转route[i:k+1]之间的路径
    """
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route


def two_opt(route, dist_matrix):
    """
    2-opt局部搜索优化
    不断尝试反转路径片段，直到无法改进为止
    """
    n = len(route)
    improved = True
    best_route = route.copy()
    best_distance = calculate_route_distance(best_route, dist_matrix)

    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                # 计算交换前后的距离变化
                # 原来的边: route[i-1]-route[i] 和 route[k]-route[k+1 mod n]
                # 新的边: route[i-1]-route[k] 和 route[i]-route[k+1 mod n]

                i_prev = i - 1
                k_next = (k + 1) % n

                # 原距离
                d1 = dist_matrix[best_route[i_prev]][best_route[i]]
                d2 = dist_matrix[best_route[k]][best_route[k_next]]

                # 新距离
                d3 = dist_matrix[best_route[i_prev]][best_route[k]]
                d4 = dist_matrix[best_route[i]][best_route[k_next]]

                # 如果新距离更短，执行交换
                if d3 + d4 < d1 + d2:
                    best_route = two_opt_swap(best_route, i, k)
                    best_distance = best_distance - (d1 + d2) + (d3 + d4)
                    improved = True

    return best_route, best_distance


def calculate_route_distance(route, dist_matrix):
    """计算路径总距离"""
    distance = 0
    for i in range(len(route)):
        distance += dist_matrix[route[i]][route[(i+1) % len(route)]]
    return distance


# ============================================================
#                    蚁群算法 + 2-opt 核心
# ============================================================

class AntColonyTSP_2opt:
    def __init__(self, dist_matrix, num_ants, num_iterations,
                 alpha, beta, rho, q, use_2opt=True, opt_2_start_iter=150,
                 opt_2_frequency=5, opt_2_for_all_ants=False):
        self.dist_matrix = dist_matrix
        self.num_cities = len(dist_matrix)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.use_2opt = use_2opt
        self.opt_2_start_iter = opt_2_start_iter
        self.opt_2_frequency = opt_2_frequency
        self.opt_2_for_all_ants = opt_2_for_all_ants

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

        # 统计2-opt改进次数
        self.opt_2_improvements = 0

        # 记录2-opt启动前的最优解（用于对比）
        self.route_before_2opt = None
        self.distance_before_2opt = None

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
        """运行蚁群算法 + 2-opt"""
        print("开始运行蚁群算法 + 2-opt...")
        print(f"城市数量: {self.num_cities}")
        print(f"蚂蚁数量: {self.num_ants}")
        print(f"迭代次数: {self.num_iterations}")
        print(f"2-opt优化: {'启用' if self.use_2opt else '禁用'}")
        if self.use_2opt:
            print(f"2-opt启动代数: 第{self.opt_2_start_iter}代")
            print(f"2-opt频率: 每{self.opt_2_frequency}代")
            print(f"全蚂蚁2-opt: {'是' if self.opt_2_for_all_ants else '否'}")
        print("-" * 40)

        for iteration in range(self.num_iterations):
            all_routes = []
            all_distances = []

            # 每只蚂蚁构建路径
            for _ in range(self.num_ants):
                route = self.construct_route()

                # 对每只蚂蚁的路径进行2-opt优化（可选，且需要达到启动代数）
                if self.use_2opt and self.opt_2_for_all_ants and iteration >= self.opt_2_start_iter:
                    route, distance = two_opt(route, self.dist_matrix)
                else:
                    distance = self.calculate_route_distance(route)

                all_routes.append(route)
                all_distances.append(distance)

                # 更新最优解
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_route = route.copy()

            # 对当前最优解进行2-opt优化（需要达到启动代数）
            if self.use_2opt and not self.opt_2_for_all_ants:
                # 在2-opt启动前一代，保存当前最优解用于对比
                if iteration == self.opt_2_start_iter - 1:
                    self.route_before_2opt = self.best_route.copy()
                    self.distance_before_2opt = self.best_distance

                if iteration >= self.opt_2_start_iter and (iteration - self.opt_2_start_iter) % self.opt_2_frequency == 0:
                    old_distance = self.best_distance
                    self.best_route, self.best_distance = two_opt(
                        self.best_route, self.dist_matrix
                    )
                    if self.best_distance < old_distance:
                        self.opt_2_improvements += 1

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
        if self.use_2opt:
            print(f"2-opt改进次数: {self.opt_2_improvements}")

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


def plot_convergence_with_params(history, runtime, num_cities, known_optimal, opt_2_improvements, output_path):
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
    # 绘制2-opt启动位置的垂直线
    if USE_2OPT and OPT_2_START_ITER < len(history):
        ax.axvline(x=OPT_2_START_ITER, color='g', linestyle=':', linewidth=2, label=f'2-opt Start (iter {OPT_2_START_ITER})')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Distance', fontsize=12)
    ax.set_title('ACO + 2-opt Convergence Curve', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 下方：参数信息区域（无背景色，水平排列）
    param_ax = fig.add_axes([0.1, 0.02, 0.85, 0.28])  # 与x轴对齐
    param_ax.axis('off')

    # 参数文本（分两列显示）
    left_text = (
        f"ACO Parameters:\n"
        f"  Ants: {NUM_ANTS}\n"
        f"  Iterations: {NUM_ITERATIONS}\n"
        f"  Alpha: {ALPHA}\n"
        f"  Beta: {BETA}\n"
        f"  Rho: {RHO}\n"
        f"  Q: {Q}\n"
        f"\n2-opt Settings:\n"
        f"  Enabled: {USE_2OPT}\n"
        f"  Start Iter: {OPT_2_START_ITER}\n"
        f"  Frequency: {OPT_2_FREQUENCY}\n"
        f"  All Ants: {OPT_2_FOR_ALL_ANTS}"
    )

    if known_optimal:
        gap = ((history[-1] - known_optimal) / known_optimal * 100)
        right_text = (
            f"Results:\n"
            f"  Cities: {num_cities}\n"
            f"  Best Distance: {history[-1]:.2f}\n"
            f"  Known Optimal: {known_optimal}\n"
            f"  Gap: {gap:.2f}%\n"
            f"  Runtime: {runtime:.2f}s\n"
            f"  2-opt Improvements: {opt_2_improvements}"
        )
    else:
        right_text = (
            f"Results:\n"
            f"  Cities: {num_cities}\n"
            f"  Best Distance: {history[-1]:.2f}\n"
            f"  Runtime: {runtime:.2f}s\n"
            f"  2-opt Improvements: {opt_2_improvements}"
        )

    # 左侧参数
    param_ax.text(0.05, 0.95, left_text, transform=param_ax.transAxes, fontsize=10,
                  verticalalignment='top', horizontalalignment='left', family='monospace')
    # 右侧结果
    param_ax.text(0.55, 0.95, right_text, transform=param_ax.transAxes, fontsize=10,
                  verticalalignment='top', horizontalalignment='left', family='monospace')

    # 添加分隔线
    param_ax.axhline(y=0.98, xmin=0.02, xmax=0.98, color='gray', linewidth=1)

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
    ax1.set_title(f'ACO+2opt Route (Distance: {current_distance:.2f})', fontsize=14)
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


def plot_2opt_effect(cities, route_before, distance_before, route_after, distance_after, output_path):
    """绘制2-opt效果对比图：左边2-opt前，右边2-opt后"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 左图：2-opt优化前的路径 ---
    ax1.scatter(cities[:, 0], cities[:, 1], c='red', s=50, zorder=5)
    for i, (x, y) in enumerate(cities):
        ax1.annotate(str(i+1), (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    if route_before:
        route_cities = cities[route_before + [route_before[0]]]
        ax1.plot(route_cities[:, 0], route_cities[:, 1], 'b-', linewidth=1.5, alpha=0.7)
        start = cities[route_before[0]]
        ax1.scatter([start[0]], [start[1]], c='green', s=150, marker='*', zorder=6, label='Start')
    ax1.set_xlabel('X Coordinate', fontsize=12)
    ax1.set_ylabel('Y Coordinate', fontsize=12)
    ax1.set_title(f'Before 2-opt (Distance: {distance_before:.2f})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 右图：2-opt优化后的路径 ---
    ax2.scatter(cities[:, 0], cities[:, 1], c='red', s=50, zorder=5)
    for i, (x, y) in enumerate(cities):
        ax2.annotate(str(i+1), (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    route_cities = cities[route_after + [route_after[0]]]
    ax2.plot(route_cities[:, 0], route_cities[:, 1], 'g-', linewidth=1.5, alpha=0.7)
    start = cities[route_after[0]]
    ax2.scatter([start[0]], [start[1]], c='green', s=150, marker='*', zorder=6, label='Start')
    ax2.set_xlabel('X Coordinate', fontsize=12)
    ax2.set_ylabel('Y Coordinate', fontsize=12)

    # 计算改进幅度
    if distance_before and distance_before > 0:
        improvement = ((distance_before - distance_after) / distance_before) * 100
        ax2.set_title(f'After 2-opt (Distance: {distance_after:.2f}, -{improvement:.2f}%)', fontsize=14)
    else:
        ax2.set_title(f'After 2-opt (Distance: {distance_after:.2f})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI)
    print(f"2-opt效果对比图已保存: {output_path}")
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

    # 创建蚁群算法+2-opt实例
    aco = AntColonyTSP_2opt(
        dist_matrix=dist_matrix,
        num_ants=NUM_ANTS,
        num_iterations=NUM_ITERATIONS,
        alpha=ALPHA,
        beta=BETA,
        rho=RHO,
        q=Q,
        use_2opt=USE_2OPT,
        opt_2_start_iter=OPT_2_START_ITER,
        opt_2_frequency=OPT_2_FREQUENCY,
        opt_2_for_all_ants=OPT_2_FOR_ALL_ANTS
    )

    # 运行算法并计时
    start_time = time.time()
    best_route, best_distance = aco.run()
    runtime = time.time() - start_time

    # 生成输出文件路径
    convergence_path = os.path.join(run_dir, "convergence.png")
    routes_path = os.path.join(run_dir, "routes_comparison.png")
    opt_effect_path = os.path.join(run_dir, "2opt_effect.png")

    # 绘制收敛曲线图（包含参数信息）
    plot_convergence_with_params(aco.convergence_history, runtime, len(cities),
                                  known_optimal, aco.opt_2_improvements, convergence_path)

    # 绘制路径对比图（本次运行 vs 已知最优）
    plot_routes_comparison(cities, best_route, best_distance,
                          optimal_tour, known_optimal, routes_path)

    # 绘制2-opt效果对比图（2-opt前 vs 2-opt后）
    if USE_2OPT and aco.route_before_2opt is not None:
        plot_2opt_effect(cities, aco.route_before_2opt, aco.distance_before_2opt,
                        best_route, best_distance, opt_effect_path)

    print("")
    print(f"第 {run_number} 次运行完成!")
    print(f"输出目录: {run_dir}")


if __name__ == "__main__":
    main()
