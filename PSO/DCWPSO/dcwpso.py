"""
DCWPSO - Dynamic Clustering Weight Particle Swarm Optimization
动态聚类权重粒子群优化算法 Python实现
"""

import numpy as np
from typing import Callable, Tuple


class DCWPSO:
    """动态聚类权重粒子群优化算法"""

    def __init__(self, fitness_func: Callable, dimension: int,
                 particle_number: int, max_fes: int,
                 lb: float, ub: float, k_neighbors: int = 2):
        """
        参数:
            fitness_func: 适应度函数
            dimension: 问题维度
            particle_number: 粒子数量
            max_fes: 最大函数评估次数
            lb: 搜索空间下界
            ub: 搜索空间上界
            k_neighbors: 邻居数量
        """
        self.fitness_func = fitness_func
        self.D = dimension
        self.NP = particle_number
        self.max_fes = max_fes
        self.k = k_neighbors

        # 边界
        self.lb = np.ones(self.D) * lb if isinstance(lb, (int, float)) else np.array(lb)
        self.ub = np.ones(self.D) * ub if isinstance(ub, (int, float)) else np.array(ub)

        # 参数
        self.cc = np.array([2.0, 2.0])
        self.later_phase = max_fes * 0.8
        self.worst_size = particle_number - 1

        # 状态变量
        self.FES = 0
        self.convergence_curve = []

    def initialize(self):
        """初始化种群、速度和个体历史最优"""
        # 初始化种群位置
        self.pop = self.lb + np.random.rand(self.NP, self.D) * (self.ub - self.lb)

        # 初始化速度
        mv = 0.1 * (self.ub - self.lb)
        self.Vmin = np.tile(-mv, (self.NP, 1))
        self.Vmax = -self.Vmin
        self.vel = self.Vmin + 2 * self.Vmax * np.random.rand(self.NP, self.D)

        # 计算初始适应度
        fitness = self._evaluate_population(self.pop)
        self.FES = self.NP

        # 初始化pbest和gbest
        self.pbest = self.pop.copy()
        self.pbestval = fitness.copy()

        # 找到全局最优
        best_idx = np.argmin(self.pbestval)
        self.gbestval = self.pbestval[best_idx]
        self.gbest = self.pbest[best_idx].copy()

        # 记录收敛曲线
        self.convergence_curve = [self.gbestval] * self.FES

        # 初始化最差个体
        sorted_indices = np.argsort(self.pbestval)
        self.worst_pop = np.array([
            self.pbest[sorted_indices[-1]],
            self.pbest[sorted_indices[-2]]
        ])

    def _evaluate_population(self, pop: np.ndarray) -> np.ndarray:
        """评估种群适应度"""
        fitness = np.array([self.fitness_func(ind) for ind in pop])
        return fitness

    def _find_gbest_neighbors(self) -> np.ndarray:
        """找到gbest的邻居粒子"""
        gbestrep = np.tile(self.gbest, (self.NP, 1))
        distances = np.linalg.norm(self.pop - gbestrep, axis=1)
        non_zero_indices = np.where(distances > 0)[0]

        if len(non_zero_indices) == 0:
            return gbestrep[:self.k]

        current_k = min(self.k, len(non_zero_indices))
        sorted_indices = np.argsort(distances[non_zero_indices])[:current_k]
        return self.pop[non_zero_indices[sorted_indices]]

    def _find_pbest_neighbors(self, particle_idx: int) -> np.ndarray:
        """找到某个粒子的pbest邻居"""
        distances = np.linalg.norm(self.pop[particle_idx] - self.pbest, axis=1)
        distances[particle_idx] = 0  # 排除自己
        non_zero_indices = np.where(distances > 0)[0]

        if len(non_zero_indices) == 0:
            return self.pbest[:self.k]

        current_k = min(self.k, len(non_zero_indices))
        sorted_indices = np.argsort(distances[non_zero_indices])[:current_k]
        return self.pop[non_zero_indices[sorted_indices]]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def optimize(self):
        """执行DCWPSO优化"""
        self.initialize()
        g = 0

        while self.FES < self.max_fes:
            g += 1

            # 动态惯性权重
            iwt = (np.random.rand() * (self.max_fes - self.FES)**2 / self.max_fes**2) * (0.9 - 0.4) + 0.4

            # 找到gbest的邻居
            gbest_neighbor = self._find_gbest_neighbors()

            # 更新每个粒子的速度
            self._update_velocities(iwt, gbest_neighbor)

            # 位置更新和边界处理
            self._update_positions()

            # 评估新位置
            fitness = self._evaluate_population(self.pop)
            self.FES += self.NP

            if self.FES > self.max_fes:
                break

            # 更新pbest和gbest
            self._update_best(fitness)

            # 记录收敛曲线
            self.convergence_curve.extend([self.gbestval] * self.NP)

        return self.gbest, self.gbestval, self.convergence_curve[:self.max_fes]

    def _update_velocities(self, iwt: float, gbest_neighbor: np.ndarray):
        """更新粒子速度"""
        gbestrep = np.tile(self.gbest, (self.NP, 1))
        ws = 0  # worst个体索引

        for i in range(self.NP):
            # 找到当前粒子的pbest邻居
            pbest_neighbor = self._find_pbest_neighbors(i)

            # 随机选择一个gbest邻居和pbest邻居
            rand_gbest_idx = np.random.randint(0, len(gbest_neighbor))
            rand_pbest_idx = np.random.randint(0, len(pbest_neighbor))

            Pgbest = gbest_neighbor[rand_gbest_idx]
            Ppbest = pbest_neighbor[rand_pbest_idx]

            # 计算余弦相似度
            cosine_simi = self._cosine_similarity(Pgbest, Ppbest)

            # 最差个体学习策略（后期）
            if (self.FES >= self.later_phase) and (i >= self.worst_size):
                self.vel[i] = iwt * self.vel[i] + self.cc[1] * np.random.rand(self.D) * (self.gbest - self.worst_pop[ws])
                ws = (ws + 1) % 2  # 在0和1之间切换
            else:
                # 根据余弦相似度选择策略
                if cosine_simi < 0 or cosine_simi < 0.5:
                    # 邻域学习策略
                    aa = 2.2 * np.random.rand(self.D) * (Ppbest - self.pop[i]) + \
                         1.8 * np.random.rand(self.D) * (Pgbest - self.pop[i])
                    self.vel[i] = iwt * self.vel[i] + aa
                else:
                    # 标准PSO策略
                    aa = self.cc[0] * np.random.rand(self.D) * (self.pbest[i] - self.pop[i]) + \
                         self.cc[1] * np.random.rand(self.D) * (gbestrep[i] - self.pop[i])
                    self.vel[i] = iwt * self.vel[i] + aa

    def _update_positions(self):
        """更新粒子位置并进行边界处理"""
        # 速度边界约束
        self.vel = np.where(self.vel > self.Vmax, self.Vmax, self.vel)
        self.vel = np.where(self.vel < self.Vmin, self.Vmin, self.vel)

        # 位置更新
        self.pop = self.pop + self.vel

        # 边界处理（两种策略随机选择）
        if np.random.rand() > 0.5:
            # 策略1：直接截断
            self.pop = np.where(self.pop > self.ub, self.ub, self.pop)
            self.pop = np.where(self.pop < self.lb, self.lb, self.pop)
        else:
            # 策略2：随机重置
            mask_lower = self.pop < self.lb
            mask_upper = self.pop > self.ub
            mask_valid = (self.pop >= self.lb) & (self.pop <= self.ub)

            self.pop = mask_valid * self.pop + \
                       mask_lower * (self.lb + 0.2 * (self.ub - self.lb) * np.random.rand(self.NP, self.D)) + \
                       mask_upper * (self.ub - 0.2 * (self.ub - self.lb) * np.random.rand(self.NP, self.D))

    def _update_best(self, fitness: np.ndarray):
        """更新个体最优和全局最优"""
        # 更新pbest
        improved = self.pbestval > fitness
        self.pbest[improved] = self.pop[improved]
        self.pbestval[improved] = fitness[improved]

        # 更新gbest
        best_idx = np.argmin(self.pbestval)
        if self.pbestval[best_idx] < self.gbestval:
            self.gbestval = self.pbestval[best_idx]
            self.gbest = self.pbest[best_idx].copy()

        # 更新最差个体
        sorted_indices = np.argsort(self.pbestval)
        self.worst_pop = np.array([
            self.pbest[sorted_indices[-1]],
            self.pbest[sorted_indices[-2]]
        ])
