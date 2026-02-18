"""
模拟退火算法变种性能对比实验
主程序入口 - 运行所有算法在所有测试函数上的对比实验
结果保存到 output/ 目录
"""
import os
import json
import time
import numpy as np
from datetime import datetime
from scipy.stats import ranksums

from benchmarks import get_all_benchmarks
from algorithms import get_all_algorithms
from visualization import (
    plot_convergence_single,
    plot_convergence_overview,
    plot_statistics_table,
)

# ============ 实验参数配置 ============
DIM = 30              # 问题维度
MAX_FES = 100000      # 最大函数评估次数
N_RUNS = 10           # 独立运行次数
SEED_BASE = 42        # 随机种子基数

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single_experiment(func, algorithms, base_seed):
    """在单个测试函数上运行所有算法一次，每个算法使用独立随机种子"""
    results = {}
    for i, algo in enumerate(algorithms):
        np.random.seed(base_seed * 100 + i)
        algo.optimize(func)
        results[algo.name] = {
            "fes": list(algo.fes_history),
            "fitness": list(algo.fitness_history),
            "best_f": algo.best_f,
        }
    return results


def interpolate_to_common_fes(fes_list, fitness_list, common_fes):
    """将不同长度的FES历史插值到统一的FES网格上"""
    fes_arr = np.array(fes_list, dtype=float)
    fit_arr = np.array(fitness_list, dtype=float)
    return np.interp(common_fes, fes_arr, fit_arr)


def main():
    print("=" * 60)
    print("  模拟退火算法变种 性能收敛对比实验")
    print(f"  维度: {DIM}  |  MaxFES: {MAX_FES}  |  独立运行: {N_RUNS}次")
    print("=" * 60)

    benchmarks = get_all_benchmarks(DIM)
    common_fes = np.linspace(1, MAX_FES, 500)

    # 存储所有结果
    all_convergence = {}       # {func_name: {algo: {fes, fitness}}}
    all_std_convergence = {}   # {func_name: {algo: {fes, std}}}
    all_stats = {}             # {func_name: {algo: {mean, std, best, worst}}}
    raw_results = {}           # 原始数据用于JSON保存

    total_start = time.time()

    for func in benchmarks:
        print(f"\n>>> 测试函数: {func.name} (bounds={func.bounds})")
        func_results_all_runs = {}  # {algo_name: [run1_data, run2_data, ...]}
        best_values = {}            # {algo_name: [best_f_run1, ...]}
        func_std_convergence = {}   # {algo_name: {"fes": [...], "std": [...]}}

        for run in range(N_RUNS):
            seed = SEED_BASE + run
            algorithms = get_all_algorithms(MAX_FES, DIM, func.bounds)
            results = run_single_experiment(func, algorithms, base_seed=seed)

            for algo_name, data in results.items():
                if algo_name not in func_results_all_runs:
                    func_results_all_runs[algo_name] = []
                    best_values[algo_name] = []
                func_results_all_runs[algo_name].append(data)
                best_values[algo_name].append(data["best_f"])

            print(f"    Run {run + 1}/{N_RUNS} done")

        # 计算统计量
        func_stats = {}
        func_avg_convergence = {}

        for algo_name in func_results_all_runs:
            vals = np.array(best_values[algo_name])
            func_stats[algo_name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "median": float(np.median(vals)),
                "best": float(np.min(vals)),
                "worst": float(np.max(vals)),
            }

            # 计算平均收敛曲线及标准差（插值到统一FES网格）
            interp_curves = []
            for run_data in func_results_all_runs[algo_name]:
                curve = interpolate_to_common_fes(
                    run_data["fes"], run_data["fitness"], common_fes
                )
                interp_curves.append(curve)
            avg_curve = np.mean(interp_curves, axis=0)
            std_curve = np.std(interp_curves, axis=0, ddof=1)
            func_avg_convergence[algo_name] = {
                "fes": common_fes.tolist(),
                "fitness": avg_curve.tolist(),
            }
            func_std_convergence[algo_name] = {
                "fes": common_fes.tolist(),
                "std": std_curve.tolist(),
            }

            print(f"    {algo_name}: "
                  f"Mean={func_stats[algo_name]['mean']:.4e}, "
                  f"Std={func_stats[algo_name]['std']:.4e}, "
                  f"Median={func_stats[algo_name]['median']:.4e}")

        # Wilcoxon 秩和检验（两两比较）
        algo_list = list(func_results_all_runs.keys())
        if len(algo_list) >= 2:
            print(f"    --- Wilcoxon Rank-Sum Test (alpha=0.05) ---")
            for i in range(len(algo_list)):
                for j in range(i + 1, len(algo_list)):
                    a_vals = np.array(best_values[algo_list[i]])
                    b_vals = np.array(best_values[algo_list[j]])
                    _, p_val = ranksums(a_vals, b_vals)
                    sig = "***" if p_val < 0.001 else (
                          "**" if p_val < 0.01 else (
                          "*" if p_val < 0.05 else "n.s."))
                    print(f"    {algo_list[i]} vs {algo_list[j]}: "
                          f"p={p_val:.4e} {sig}")

        all_convergence[func.name] = func_avg_convergence
        all_std_convergence[func.name] = func_std_convergence
        all_stats[func.name] = func_stats
        raw_results[func.name] = {
            "stats": func_stats,
            "convergence": func_avg_convergence,
        }

        # 绘制单函数收敛曲线（含置信区间）
        p = plot_convergence_single(
            func_avg_convergence, func.name, DIM, OUTPUT_DIR,
            std_curves=func_std_convergence,
        )
        print(f"    -> 收敛曲线已保存: {p}")

    # ---- 绘制总览图 ----
    print("\n>>> 绘制总览对比图...")
    p = plot_convergence_overview(all_convergence, DIM, OUTPUT_DIR,
                                  all_std_curves=all_std_convergence)
    print(f"    -> 总览图已保存: {p}")

    # ---- 绘制统计表格 ----
    print(">>> 绘制统计结果表格...")
    p = plot_statistics_table(all_stats, OUTPUT_DIR)
    print(f"    -> 统计表格已保存: {p}")

    # ---- 保存原始数据为JSON ----
    json_path = os.path.join(OUTPUT_DIR, "results.json")
    report = {
        "experiment_info": {
            "dim": DIM,
            "max_fes": MAX_FES,
            "n_runs": N_RUNS,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": raw_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n>>> 原始数据已保存: {json_path}")

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  实验完成! 总耗时: {elapsed:.1f}s")
    print(f"  所有结果已保存到: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
