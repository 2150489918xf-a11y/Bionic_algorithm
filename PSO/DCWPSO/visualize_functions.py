# -*- coding: utf-8 -*-
"""
基准测试函数可视化
Visualization of Benchmark Functions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 定义测试函数 ====================

def sphere(x, y):
    """Sphere函数 - 单峰凸函数"""
    return x**2 + y**2


def rastrigin(x, y):
    """Rastrigin函数 - 多峰函数"""
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)


def rosenbrock(x, y):
    """Rosenbrock函数 - 山谷型函数"""
    return 100*(y - x**2)**2 + (1 - x)**2


def ackley(x, y):
    """Ackley函数 - 多峰函数"""
    return -20*np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + 20 + np.e


def griewank(x, y):
    """Griewank函数 - 多峰函数"""
    return 1 + (x**2 + y**2)/4000 - np.cos(x)*np.cos(y/np.sqrt(2))


# ==================== 绘制函数 ====================

def plot_function_3d(func, func_name, x_range, y_range, subplot_idx):
    """绘制3D曲面图"""
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    ax = plt.subplot(2, 3, subplot_idx, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    ax.set_title(f'{func_name}', fontsize=11, pad=10)
    ax.view_init(elev=30, azim=45)

    return surf


def plot_function_contour(func, func_name, x_range, y_range, subplot_idx):
    """绘制等高线图"""
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    ax = plt.subplot(2, 3, subplot_idx)
    contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{func_name}', fontsize=11)
    ax.plot(0, 0, 'r*', markersize=10, label='全局最优点')
    ax.legend(fontsize=8)

    return contour


# ==================== 主函数 ====================

def main():
    """主函数：绘制所有测试函数"""

    # 定义测试函数配置
    functions = [
        (sphere, "Sphere函数 (单峰凸函数)", (-5, 5), (-5, 5)),
        (rastrigin, "Rastrigin函数 (多峰函数)", (-5, 5), (-5, 5)),
        (rosenbrock, "Rosenbrock函数 (山谷型)", (-2, 2), (-1, 3)),
        (ackley, "Ackley函数 (多峰函数)", (-5, 5), (-5, 5)),
        (griewank, "Griewank函数 (多峰函数)", (-10, 10), (-10, 10))
    ]

    # 绘制3D曲面图
    print("正在生成3D曲面图...")
    fig1 = plt.figure(figsize=(15, 10))
    fig1.suptitle('基准测试函数 - 3D曲面图', fontsize=16, fontweight='bold')

    for i, (func, name, x_range, y_range) in enumerate(functions, 1):
        plot_function_3d(func, name, x_range, y_range, i)

    plt.tight_layout()
    plt.savefig('benchmark_functions_3d.png', dpi=300, bbox_inches='tight')
    print("3D曲面图已保存: benchmark_functions_3d.png")

    # 绘制等高线图
    print("正在生成等高线图...")
    fig2 = plt.figure(figsize=(15, 10))
    fig2.suptitle('基准测试函数 - 等高线图', fontsize=16, fontweight='bold')

    for i, (func, name, x_range, y_range) in enumerate(functions, 1):
        plot_function_contour(func, name, x_range, y_range, i)

    plt.tight_layout()
    plt.savefig('benchmark_functions_contour.png', dpi=300, bbox_inches='tight')
    print("等高线图已保存: benchmark_functions_contour.png")

    # plt.show()  # 注释掉交互式显示，避免阻塞
    print("\n可视化完成！")
    print("生成的图片:")
    print("  - benchmark_functions_3d.png (3D曲面图)")
    print("  - benchmark_functions_contour.png (等高线图)")


def print_function_info():
    """打印函数详细信息"""
    print("="*70)
    print("基准测试函数详细说明")
    print("="*70)

    info = """
1. Sphere函数 (单峰凸函数)
   - 公式: f(x) = sum(xi^2)
   - 特点: 最简单的凸函数，全局最优点在原点
   - 最优值: f(0,0,...,0) = 0
   - 难度: ★☆☆☆☆ (最简单)

2. Rastrigin函数 (多峰函数)
   - 公式: f(x) = 10n + sum[xi^2 - 10cos(2πxi)]
   - 特点: 大量局部最优点，容易陷入局部最优
   - 最优值: f(0,0,...,0) = 0
   - 难度: ★★★★☆ (困难)

3. Rosenbrock函数 (山谷型函数)
   - 公式: f(x,y) = 100(y-x^2)^2 + (1-x)^2
   - 特点: 狭长的抛物线山谷，全局最优在山谷底部
   - 最优值: f(1,1,...,1) = 0
   - 难度: ★★★☆☆ (中等)

4. Ackley函数 (多峰函数)
   - 公式: f(x) = -20exp(-0.2*sqrt(sum(xi^2)/n)) - exp(sum(cos(2πxi))/n) + 20 + e
   - 特点: 中心有深谷，周围有大量局部最优
   - 最优值: f(0,0,...,0) = 0
   - 难度: ★★★★☆ (困难)

5. Griewank函数 (多峰函数)
   - 公式: f(x) = 1 + sum(xi^2/4000) - prod[cos(xi/sqrt(i))]
   - 特点: 大量局部最优，但随维度增加难度降低
   - 最优值: f(0,0,...,0) = 0
   - 难度: ★★★☆☆ (中等)
    """
    print(info)
    print("="*70)


if __name__ == "__main__":
    print_function_info()
    main()

