"""
基于FES的算法性能收敛对比可视化模块
使用matplotlib绘制收敛曲线图
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 配置matplotlib支持中文显示
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


# 算法对应的绘图样式
ALGO_STYLES = {
    "Classical SA":            {"color": "#E74C3C", "ls": "-",  "marker": "o"},
    "Adaptive SA (ASA)":       {"color": "#2ECC71", "ls": "--", "marker": "s"},
    "Restarting SA (RSA)":     {"color": "#3498DB", "ls": "-.", "marker": "^"},
    "Simulated Quenching (SQA)": {"color": "#F39C12", "ls": ":", "marker": "D"},
    "Curious SA (CSA)":        {"color": "#9B59B6", "ls": "-",  "marker": "p"},
}


def plot_convergence_single(results, func_name, dim, output_dir,
                            std_curves=None):
    """
    绘制单个测试函数上各算法的FES收敛曲线
    results: dict {algo_name: {"fes": [...], "fitness": [...]}}
    std_curves: dict {algo_name: {"fes": [...], "std": [...]}} 可选，用于绘制置信区间
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo_name, data in results.items():
        style = ALGO_STYLES.get(algo_name, {"color": "gray", "ls": "-", "marker": "x"})
        fes = np.array(data["fes"])
        fitness = np.array(data["fitness"])

        # 每隔一定间距标记点，避免过密
        n_markers = min(15, len(fes))
        marker_every = max(1, len(fes) // n_markers)

        ax.plot(
            fes, fitness,
            label=algo_name,
            color=style["color"],
            linestyle=style["ls"],
            marker=style["marker"],
            markevery=marker_every,
            markersize=5,
            linewidth=1.8,
            alpha=0.9,
        )


    ax.set_xlabel("FES (Function Evaluations)", fontsize=12)
    ax.set_ylabel("Best Fitness (log scale)", fontsize=12)
    ax.set_title(f"Convergence Comparison — {func_name} (D={dim})", fontsize=14)
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()

    path = os.path.join(output_dir, f"convergence_{func_name}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_convergence_overview(all_results, dim, output_dir,
                              all_std_curves=None):
    """
    绘制所有测试函数的收敛曲线总览图（子图网格）
    all_results: dict {func_name: {algo_name: {"fes": [...], "fitness": [...]}}}
    all_std_curves: dict {func_name: {algo_name: {"fes": [...], "std": [...]}}} 可选
    """
    func_names = list(all_results.keys())
    n = len(func_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, func_name in enumerate(func_names):
        ax = axes[idx]
        results = all_results[func_name]
        std_curves = (all_std_curves or {}).get(func_name, None)

        for algo_name, data in results.items():
            style = ALGO_STYLES.get(
                algo_name, {"color": "gray", "ls": "-", "marker": "x"}
            )
            fes = np.array(data["fes"])
            fitness = np.array(data["fitness"])
            n_markers = min(10, len(fes))
            marker_every = max(1, len(fes) // n_markers)

            ax.plot(
                fes, fitness,
                label=algo_name,
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markevery=marker_every,
                markersize=4,
                linewidth=1.5,
                alpha=0.85,
            )


        ax.set_xlabel("FES", fontsize=9)
        ax.set_ylabel("Best Fitness", fontsize=9)
        ax.set_title(func_name, fontsize=11, fontweight="bold")
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=8)

    # 统一图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4, fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    # 隐藏多余子图
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"SA Variants Convergence Comparison (D={dim})",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "convergence_overview.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_statistics_table(stats, output_dir):
    """
    绘制多次独立运行的统计结果表格图
    stats: dict {func_name: {algo_name: {"mean","std","median","best","worst"}}}
    每行自动高亮 Mean 最小的算法（浅绿色背景 + 加粗）
    """
    func_names = list(stats.keys())
    algo_names = list(next(iter(stats.values())).keys())

    fig, ax = plt.subplots(
        figsize=(3 + 3.0 * len(algo_names), 1.2 + 0.45 * len(func_names))
    )
    ax.axis("off")

    # 构建表格数据（增加 Median 列）
    col_labels = ["Function"]
    for algo in algo_names:
        col_labels.extend([f"{algo}\nMean", f"{algo}\nStd", f"{algo}\nMedian"])

    table_data = []
    best_col_indices = []  # 每行中 Mean 最小的算法对应的列索引
    for func_name in func_names:
        row = [func_name]
        means = []
        mean_col_indices = []
        for algo in algo_names:
            s = stats[func_name][algo]
            col_idx = len(row)
            mean_col_indices.append(col_idx)
            means.append(s["mean"])
            row.append(f"{s['mean']:.4e}")
            row.append(f"{s['std']:.4e}")
            row.append(f"{s.get('median', s['mean']):.4e}")
        # 找到 Mean 最小的列索引
        best_idx = int(np.argmin(means))
        best_col_indices.append(mean_col_indices[best_idx])
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # 表头样式
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#34495E")
        cell.set_text_props(color="white", fontweight="bold")

    # 交替行颜色 + 最优高亮
    for i in range(len(table_data)):
        base_color = "#ECF0F1" if i % 2 == 0 else "#FFFFFF"
        best_col = best_col_indices[i]
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            # 最优算法的 Mean/Std/Median 三列高亮
            if best_col <= j <= best_col + 2:
                cell.set_facecolor("#A9DFBF")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor(base_color)

    ax.set_title(
        "Statistical Results (Multiple Independent Runs)",
        fontsize=13, fontweight="bold", pad=20,
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "statistics_table.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
