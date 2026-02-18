"""
模拟退火算法变种集合
包含: 经典SA, 自适应ASA, 重启RSA, 模拟淬火SQA
所有算法统一以FES(Function Evaluations)作为计算预算度量
"""
import numpy as np
from abc import ABC, abstractmethod


class SABase(ABC):
    """模拟退火算法基类"""

    def __init__(self, name, max_fes, dim, bounds):
        self.name = name
        self.max_fes = max_fes
        self.dim = dim
        lb, ub = bounds
        # 支持向量化 bounds：标量自动广播为数组
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.asarray(lb)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.asarray(ub)
        self.range = self.ub - self.lb  # 预计算搜索范围
        self.fes = 0
        self.best_x = None
        self.best_f = np.inf
        self.fes_history = []
        self.fitness_history = []

    def _init_solution(self):
        return np.random.uniform(self.lb, self.ub, self.dim)

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def _evaluate(self, func, x):
        f = func(x)
        self.fes += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        if (self.fes <= 10 or
            self.fes % max(1, self.max_fes // 500) == 0 or
            self.fes == self.max_fes):
            self.fes_history.append(self.fes)
            self.fitness_history.append(self.best_f)
        return f

    def _accept(self, delta, T):
        """安全的 Metropolis 接受判断，避免 exp 溢出"""
        if delta < 0:
            return True
        ratio = delta / max(T, 1e-30)
        if ratio > 500:
            return False
        return np.random.rand() < np.exp(-ratio)

    def _record_final(self):
        if not self.fes_history or self.fes_history[-1] != self.fes:
            self.fes_history.append(self.fes)
            self.fitness_history.append(self.best_f)

    def reset(self):
        self.fes = 0
        self.best_x = None
        self.best_f = np.inf
        self.fes_history = []
        self.fitness_history = []

    @abstractmethod
    def optimize(self, func):
        pass


# ============================================================
# 1. 经典模拟退火 (Classical Simulated Annealing)
# ============================================================
class ClassicalSA(SABase):
    """
    经典模拟退火算法
    - 指数冷却: T(k+1) = alpha * T(k)
    - Metropolis接受准则
    - 高斯邻域扰动
    """

    def __init__(self, max_fes, dim, bounds,
                 T0=1000, T_min=1e-10, alpha=0.95, markov_len=None):
        super().__init__("Classical SA", max_fes, dim, bounds)
        self.T0 = T0
        self.T_min = T_min
        self.alpha = alpha
        self.markov_len = markov_len or max(50, dim)

    def optimize(self, func):
        self.reset()
        x = self._init_solution()
        fx = self._evaluate(func, x)
        T = self.T0

        while self.fes < self.max_fes and T > self.T_min:
            for _ in range(self.markov_len):
                if self.fes >= self.max_fes:
                    break
                sigma = self.range * 0.1 * (T / self.T0)
                x_new = self._clip(x + np.random.normal(0, sigma, self.dim))
                fx_new = self._evaluate(func, x_new)
                delta = fx_new - fx
                if self._accept(delta, T):
                    x = x_new
                    fx = fx_new
            T *= self.alpha

        self._record_final()
        return self.best_x, self.best_f


# ============================================================
# 2. 自适应模拟退火 (Adaptive Simulated Annealing, ASA)
# ============================================================
class AdaptiveSA(SABase):
    """
    自适应模拟退火 (Ingber, 1989)
    - 每个维度独立温度参数
    - Cauchy分布自适应步长
    - 温度指数-指数衰减: T_i(k) = T0_i * exp(-c_i * k^(1/dim))
    """

    def __init__(self, max_fes, dim, bounds, T0=1000, T_min=1e-10):
        super().__init__("Adaptive SA (ASA)", max_fes, dim, bounds)
        self.T0 = T0
        self.T_min = T_min

    def _cauchy_perturbation(self, x, temps):
        u = np.random.uniform(-1, 1, self.dim)
        safe_temps = np.maximum(temps, 1e-30)
        exponent = np.abs(2 * u - 1)
        # 用 exp(exponent * log(1 + 1/t)) 避免直接幂运算溢出
        log_base = np.log1p(1.0 / safe_temps)
        y = np.sign(u) * safe_temps * (np.exp(np.clip(exponent * log_base, 0, 500)) - 1)
        x_new = x + y * self.range
        return self._clip(x_new)

    def optimize(self, func):
        self.reset()
        x = self._init_solution()
        fx = self._evaluate(func, x)
        temps = np.full(self.dim, float(self.T0))
        c = np.exp(-np.ones(self.dim) * 5.0)
        k = 0

        while self.fes < self.max_fes and np.max(temps) > self.T_min:
            k += 1
            x_new = self._cauchy_perturbation(x, temps)
            fx_new = self._evaluate(func, x_new)
            delta = fx_new - fx
            T_accept = np.mean(temps)
            if self._accept(delta, T_accept):
                x = x_new
                fx = fx_new
            temps = self.T0 * np.exp(-c * (k ** (1.0 / self.dim)))
            temps = np.maximum(temps, self.T_min)

        self._record_final()
        return self.best_x, self.best_f


# ============================================================
# 3. 重启模拟退火 (Restarting Simulated Annealing, RSA)
# ============================================================
class RestartingSA(SABase):
    """
    重启模拟退火
    - 温度降至阈值时以全局最优解为中心重启
    - 每次重启缩小搜索范围，逐步精细化
    - 重启时温度重置但逐次降低初始温度
    """

    def __init__(self, max_fes, dim, bounds,
                 T0=1000, T_min=1e-10, alpha=0.93,
                 restart_threshold=1.0, markov_len=None):
        super().__init__("Restarting SA (RSA)", max_fes, dim, bounds)
        self.T0 = T0
        self.T_min = T_min
        self.alpha = alpha
        self.restart_threshold = restart_threshold
        self.markov_len = markov_len or max(50, dim)

    def optimize(self, func):
        self.reset()
        x = self._init_solution()
        fx = self._evaluate(func, x)
        restart_count = 0
        T0_current = self.T0

        while self.fes < self.max_fes:
            T = T0_current
            while self.fes < self.max_fes and T > self.restart_threshold:
                for _ in range(self.markov_len):
                    if self.fes >= self.max_fes:
                        break
                    sigma = self.range * 0.1 * (T / T0_current)
                    x_new = self._clip(x + np.random.normal(0, sigma, self.dim))
                    fx_new = self._evaluate(func, x_new)
                    delta = fx_new - fx
                    if self._accept(delta, T):
                        x = x_new
                        fx = fx_new
                T *= self.alpha

            restart_count += 1
            T0_current = self.T0 / (1 + restart_count)
            shrink = max(0.1, 1.0 / (1 + restart_count))
            range_size = self.range * shrink
            x = self._clip(
                self.best_x
                + np.random.uniform(-range_size / 2, range_size / 2, self.dim)
            )
            fx = self._evaluate(func, x)

        self._record_final()
        return self.best_x, self.best_f


# ============================================================
# 4. 模拟淬火 (Simulated Quenching Algorithm, SQA)
# ============================================================
class SimulatedQuenching(SABase):
    """
    模拟淬火算法
    - 快速冷却: T(k) = T0 / (1 + k)
    - Boltzmann分布扰动
    - 贪心局部搜索增强
    """

    def __init__(self, max_fes, dim, bounds,
                 T0=1000, T_min=1e-10, markov_len=None):
        super().__init__("Simulated Quenching (SQA)", max_fes, dim, bounds)
        self.T0 = T0
        self.T_min = T_min
        self.markov_len = markov_len or max(50, dim)

    def optimize(self, func):
        self.reset()
        x = self._init_solution()
        fx = self._evaluate(func, x)
        k = 0
        quenched = False  # 标记温度是否已触底

        while self.fes < self.max_fes:
            k += 1
            T = self.T0 / (1 + k)

            if T < self.T_min and not quenched:
                quenched = True

            if not quenched:
                # 正常退火阶段
                for _ in range(self.markov_len):
                    if self.fes >= self.max_fes:
                        break
                    sigma = self.range * 0.1 * np.sqrt(T / self.T0)
                    x_new = self._clip(
                        x + np.random.normal(0, sigma, self.dim)
                    )
                    fx_new = self._evaluate(func, x_new)
                    delta = fx_new - fx
                    if self._accept(delta, T):
                        x = x_new
                        fx = fx_new

                # 贪心局部搜索
                if self.fes < self.max_fes:
                    sigma_local = self.range * 0.01 * (T / self.T0)
                    x_local = self._clip(
                        self.best_x
                        + np.random.normal(0, sigma_local, self.dim)
                    )
                    fx_local = self._evaluate(func, x_local)
                    if fx_local < fx:
                        x = x_local
                        fx = fx_local
            else:
                # 温度触底后：纯贪心局部搜索，以best_x为中心小范围精搜
                sigma_local = self.range * 0.005
                x_local = self._clip(
                    self.best_x
                    + np.random.normal(0, sigma_local, self.dim)
                )
                fx_local = self._evaluate(func, x_local)
                if fx_local < fx:
                    x = x_local
                    fx = fx_local

        self._record_final()
        return self.best_x, self.best_f


# ============================================================
# 5. 好奇模拟退火 (Curious Simulated Annealing, CSA)
# ============================================================
class CuriousSA(SABase):
    """
    好奇模拟退火算法 (CSA)
    - 基于粒子群的重采样机制
    - FSA 快速降温调度: T_k = 1 / ((k+1) * log(k+1))
    - 广义接受函数: q(rho) = 1 / (1 + rho)
    """

    def __init__(self, max_fes, dim, bounds, n_particles=30):
        super().__init__("Curious SA (CSA)", max_fes, dim, bounds)
        self.n_particles = n_particles

    def _get_temperature(self, k):
        """FSA 快速降温调度"""
        k_eff = k + 1
        return 1.0 / (k_eff * np.log(k_eff + 1.0))

    def optimize(self, func):
        self.reset()
        # 初始化粒子群
        particles = np.array([self._init_solution() for _ in range(self.n_particles)])
        values = np.array([self._evaluate(func, p) for p in particles])

        k = 0
        while self.fes < self.max_fes:
            k += 1
            T_current = self._get_temperature(k)
            T_prev = self._get_temperature(k - 1)

            # 步骤 1: 计算权重 (Reweighting)
            inv_T_diff = (1.0 / T_current) - (1.0 / T_prev)
            log_weights = -values * inv_T_diff
            log_weights -= np.max(log_weights)  # 数值稳定
            weights = np.exp(log_weights)
            weights /= np.sum(weights)

            # 步骤 2: 重采样 (Resampling)
            indices = np.random.choice(self.n_particles, size=self.n_particles, p=weights)
            particles = particles[indices]
            values = values[indices]

            # 步骤 3: 传播与变异 (Propagation)
            for i in range(self.n_particles):
                if self.fes >= self.max_fes:
                    break
                x = particles[i]
                fx = values[i]

                # 高斯扰动
                sigma = self.range * 0.1 * np.sqrt(max(T_current, 1e-10))
                y = self._clip(x + np.random.normal(0, sigma, self.dim))
                fy = self._evaluate(func, y)

                # FSA 广义接受函数
                if fy <= fx:
                    particles[i] = y
                    values[i] = fy
                else:
                    rho = (fy - fx) / max(T_current, 1e-30)
                    accept_prob = 1.0 / (1.0 + rho)
                    if np.random.rand() < accept_prob:
                        particles[i] = y
                        values[i] = fy

        self._record_final()
        return self.best_x, self.best_f


# ============================================================
# 工厂函数
# ============================================================
def get_all_algorithms(max_fes, dim, bounds):
    """获取所有算法实例"""
    return [
        ClassicalSA(max_fes, dim, bounds),
        AdaptiveSA(max_fes, dim, bounds),
        RestartingSA(max_fes, dim, bounds),
        SimulatedQuenching(max_fes, dim, bounds),
        CuriousSA(max_fes, dim, bounds),
    ]
