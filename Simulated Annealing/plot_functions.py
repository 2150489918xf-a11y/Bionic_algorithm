"""
测试函数可视化模块
生成 5 个 benchmark 函数的 2D 等高线图和 3D 曲面图
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# 配置 matplotlib
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "function_visualization")


def sphere(x, y):
    return x**2 + y**2


def rastrigin(x, y):
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)


def ackley(x, y):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))
    return term1 + term2 + 20 + np.e


def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


def griewank(x, y):
    sum_part = (x**2 + y**2) / 4000
    prod_part = np.cos(x) * np.cos(y / np.sqrt(2))
    return sum_part - prod_part + 1


# 函数配置: (函数, 范围, 名称, 全局最优点)
FUNCTIONS = [
    (sphere, (-5, 5), "Sphere", (0, 0)),
    (rastrigin, (-5.12, 5.12), "Rastrigin", (0, 0)),
    (ackley, (-5, 5), "Ackley", (0, 0)),
    (rosenbrock, (-2, 2), "Rosenbrock", (1, 1)),
    (griewank, (-10, 10), "Griewank", (0, 0)),
]


def plot_function(func, bounds, name, optimum, output_dir):
    """绘制单个函数的 2D 等高线图和 3D 曲面图"""
    # 生成网格数据
    x = np.linspace(bounds[0], bounds[1], 200)
    y = np.linspace(bounds[0], bounds[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(14, 6))

    # 左侧: 2D 等高线图
    ax1 = fig.add_subplot(1, 2, 1)
    levels = 50
    contour = ax1.contourf(X, Y, Z, levels=levels, cmap='viridis')
    ax1.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    fig.colorbar(contour, ax=ax1, label='f(x, y)')
    ax1.plot(optimum[0], optimum[1], 'r*', markersize=15, label='Global Optimum')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'{name} - 2D Contour')
    ax1.legend()
    ax1.set_aspect('equal')

    # 右侧: 3D 曲面图
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9,
                            edgecolor='none', antialiased=True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x, y)')
    ax2.set_title(f'{name} - 3D Surface')
    ax2.view_init(elev=30, azim=45)

    fig.suptitle(f'{name} Function', fontsize=16, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(output_dir, f'{name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def main():
    """生成所有函数的可视化图片"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for func, bounds, name, optimum in FUNCTIONS:
        plot_function(func, bounds, name, optimum, OUTPUT_DIR)

    print(f"\nAll images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
