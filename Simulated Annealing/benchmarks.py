"""
Benchmark测试函数集
用于评估优化算法性能的标准测试函数
"""
import numpy as np
from abc import ABC, abstractmethod


class BenchmarkFunction(ABC):
    """测试函数抽象基类"""

    def __init__(self, name, dim, bounds, global_min):
        self.name = name
        self.dim = dim
        self.bounds = bounds  # (lower, upper)
        self.global_min = global_min

    @abstractmethod
    def __call__(self, x):
        pass


class Sphere(BenchmarkFunction):
    """Sphere函数: f(x) = sum(x_i^2), 全局最优 f(0,...,0) = 0"""

    def __init__(self, dim=30):
        super().__init__("Sphere", dim, (-100, 100), 0.0)

    def __call__(self, x):
        return np.sum(x ** 2)


class Rastrigin(BenchmarkFunction):
    """Rastrigin函数: 多峰函数, 全局最优 f(0,...,0) = 0"""

    def __init__(self, dim=30):
        super().__init__("Rastrigin", dim, (-5.12, 5.12), 0.0)

    def __call__(self, x):
        return 10 * self.dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


class Ackley(BenchmarkFunction):
    """Ackley函数: 多峰函数, 全局最优 f(0,...,0) = 0"""

    def __init__(self, dim=30):
        super().__init__("Ackley", dim, (-32, 32), 0.0)

    def __call__(self, x):
        n = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


class Rosenbrock(BenchmarkFunction):
    """Rosenbrock函数: 山谷函数, 全局最优 f(1,...,1) = 0"""

    def __init__(self, dim=30):
        super().__init__("Rosenbrock", dim, (-30, 30), 0.0)

    def __call__(self, x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


class Griewank(BenchmarkFunction):
    """Griewank函数: 多峰函数, 全局最优 f(0,...,0) = 0"""

    def __init__(self, dim=30):
        super().__init__("Griewank", dim, (-600, 600), 0.0)

    def __call__(self, x):
        sum_part = np.sum(x ** 2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_part - prod_part + 1


def get_all_benchmarks(dim=30):
    """获取所有测试函数实例"""
    return [
        Sphere(dim),
        Rastrigin(dim),
        Ackley(dim),
        Rosenbrock(dim),
        Griewank(dim),
    ]
