"""
Metropolis 准则可视化
P = exp(-ΔE / T)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "function_visualization")


def plot_metropolis():
    """绘制 P 关于 T 的函数图像，不同 ΔE 值"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    T = np.linspace(0.01, 100, 500)
    delta_E_values = [1, 5, 10, 20, 50]
    colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6']

    for dE, color in zip(delta_E_values, colors):
        P = np.exp(-dE / T)
        ax.plot(T, P, label=f'ΔE = {dE}', color=color, linewidth=2)

    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('Acceptance Probability (P)', fontsize=12)
    ax.set_title('Metropolis Criterion: P = exp(-ΔE / T)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'Metropolis_Criterion.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_metropolis()
