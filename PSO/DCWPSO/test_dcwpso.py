# -*- coding: utf-8 -*-
"""
DCWPSO Algorithm Test
Test on common benchmark functions
"""

import numpy as np
import matplotlib.pyplot as plt
from dcwpso import DCWPSO

# Set matplotlib font to support Chinese
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue


# ==================== Benchmark Functions ====================

def sphere(x):
    """Sphere function: f(x) = sum(x^2), optimal value = 0"""
    return np.sum(x**2)


def rastrigin(x):
    """Rastrigin function: multimodal, optimal value = 0"""
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    """Rosenbrock function: optimal value = 0"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def ackley(x):
    """Ackley function: multimodal, optimal value = 0"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def griewank(x):
    """Griewank function: multimodal, optimal value = 0"""
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1


# ==================== Test Functions ====================

def test_function(func, func_name, dimension, lb, ub):
    """Test a single function"""
    print(f"\n{'='*60}")
    print(f"Testing Function: {func_name}")
    print(f"Dimension: {dimension}, Search Range: [{lb}, {ub}]")
    print(f"{'='*60}")

    # Create DCWPSO optimizer
    optimizer = DCWPSO(
        fitness_func=func,
        dimension=dimension,
        particle_number=30,
        max_fes=10000,
        lb=lb,
        ub=ub,
        k_neighbors=2
    )

    # Execute optimization
    gbest, gbestval, curve = optimizer.optimize()

    print(f"\nBest Solution: {gbest}")
    print(f"Best Value: {gbestval:.6e}")
    print(f"Theoretical Optimal: 0")

    return curve


def main():
    """Main test function"""
    print("DCWPSO Algorithm Test")
    print("="*60)

    # Test configuration
    dimension = 30
    test_cases = [
        (sphere, "Sphere", -100, 100),
        (rastrigin, "Rastrigin", -5.12, 5.12),
        (rosenbrock, "Rosenbrock", -30, 30),
        (ackley, "Ackley", -32, 32),
        (griewank, "Griewank", -600, 600)
    ]

    # Store convergence curves
    curves = {}

    # Test each function
    for func, name, lb, ub in test_cases:
        curve = test_function(func, name, dimension, lb, ub)
        curves[name] = curve

    # Plot convergence curves
    plot_convergence_curves(curves)


def plot_convergence_curves(curves):
    """Plot convergence curves"""
    plt.figure(figsize=(12, 8))

    for i, (name, curve) in enumerate(curves.items(), 1):
        plt.subplot(2, 3, i)
        plt.semilogy(curve, linewidth=2)
        plt.xlabel('函数评估次数 (FES)', fontsize=10)
        plt.ylabel('适应度值 (对数尺度)', fontsize=10)
        plt.title(f'{name}函数', fontsize=12)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dcwpso_results.png', dpi=300, bbox_inches='tight')
    print(f"\nConvergence curves saved to: dcwpso_results.png")
    plt.show()


if __name__ == "__main__":
    main()
