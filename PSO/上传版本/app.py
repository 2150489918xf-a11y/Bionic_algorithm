import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 设置页面配置
st.set_page_config(layout="wide", page_title="PSO Visualization")

# ==========================================
# 1. 地形配置 (保持不变)
# ==========================================
TERRAIN_CONFIGS = {
    'Six-Hump Camel': {
        'func': lambda x, y: (4 - 2.1*x**2 + (x**4)/3)*x**2 + x*y + (-4 + 4*y**2)*y**2,
        'x_bound': [-2, 2], 'y_bound': [-1, 1], 'z_lim': [-1.5, 6]
    },
    'Rastrigin': {
        'func': lambda x, y: 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y),
        'x_bound': [-5, 5], 'y_bound': [-5, 5], 'z_lim': [0, 80]
    },
    'Ackley': {
        'func': lambda x, y: -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.e + 20,
        'x_bound': [-4, 4], 'y_bound': [-4, 4], 'z_lim': [0, 15]
    },
     'Sphere': {
        'func': lambda x, y: x**2 + y**2,
        'x_bound': [-3, 3], 'y_bound': [-3, 3], 'z_lim': [0, 18]
    }
}

# ==========================================
# 2. 侧边栏控制
# ==========================================
st.sidebar.header("控制面板")
selected_terrain = st.sidebar.selectbox("选择地形", list(TERRAIN_CONFIGS.keys()))
num_particles = st.sidebar.slider("粒子数量", 10, 100, 25)
steps = st.sidebar.slider("迭代次数", 50, 200, 120)

config = TERRAIN_CONFIGS[selected_terrain]

# ==========================================
# 3. PSO 计算逻辑 (缓存以提高性能)
# ==========================================
@st.cache_data
def run_pso(terrain_name, n_particles, n_steps):
    cfg = TERRAIN_CONFIGS[terrain_name]
    func = cfg['func']
    x_bound, y_bound = cfg['x_bound'], cfg['y_bound']
    
    X = np.random.uniform(x_bound[0], x_bound[1], n_particles)
    Y = np.random.uniform(y_bound[0], y_bound[1], n_particles)
    
    v_max_x = (x_bound[1] - x_bound[0]) * 0.02
    v_max_y = (y_bound[1] - y_bound[0]) * 0.02
    V_x = np.random.uniform(-v_max_x, v_max_x, n_particles)
    V_y = np.random.uniform(-v_max_y, v_max_y, n_particles)
    
    pbest_x, pbest_y = X.copy(), Y.copy()
    pbest_z = func(X, Y)
    gbest_idx = np.argmin(pbest_z)
    gbest_x, gbest_y = pbest_x[gbest_idx], pbest_y[gbest_idx]
    
    history = []
    
    w, c1, c2 = 0.85, 0.4, 0.8
    
    for _ in range(n_steps):
        r1, r2 = np.random.rand(n_particles), np.random.rand(n_particles)
        V_x = w * V_x + c1 * r1 * (pbest_x - X) + c2 * r2 * (gbest_x - X)
        V_y = w * V_y + c1 * r1 * (pbest_y - Y) + c2 * r2 * (gbest_y - Y)
        
        X = np.clip(X + V_x, x_bound[0], x_bound[1])
        Y = np.clip(Y + V_y, y_bound[0], y_bound[1])
        current_z = func(X, Y)
        
        mask = current_z < pbest_z
        pbest_x[mask], pbest_y[mask], pbest_z[mask] = X[mask], Y[mask], current_z[mask]
        
        if np.min(pbest_z) < pbest_z[gbest_idx]:
            gbest_idx = np.argmin(pbest_z)
            gbest_x, gbest_y = pbest_x[gbest_idx], pbest_y[gbest_idx]
            
        history.append((X.copy(), Y.copy(), current_z.copy(), gbest_x, gbest_y, pbest_z[gbest_idx]))
        
    return history

# 运行计算
history = run_pso(selected_terrain, num_particles, steps)

# ==========================================
# 4. 交互式绘图
# ==========================================
st.title(f"PSO 优化可视化: {selected_terrain}")

# 使用 Slider 控制帧
frame_idx = st.slider("拖动滑块查看迭代过程", 0, len(history)-1, 0)
hx, hy, hz, gx, gy, gz = history[frame_idx]

# 创建图表
fig = plt.figure(figsize=(12, 6))

# 3D Plot
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
x_grid = np.linspace(config['x_bound'][0], config['x_bound'][1], 30)
y_grid = np.linspace(config['y_bound'][0], config['y_bound'][1], 30)
X_g, Y_g = np.meshgrid(x_grid, y_grid)
Z_g = config['func'](X_g, Y_g)

ax3d.plot_surface(X_g, Y_g, Z_g, cmap='GnBu', alpha=0.5)
ax3d.scatter(hx, hy, hz, c='red', s=30, label='Particles')
ax3d.scatter([gx], [gy], [gz], c='gold', s=100, marker='*', label='Global Best')
ax3d.set_title("3D View")

# 2D Plot
ax2d = fig.add_subplot(1, 2, 2)
ax2d.contourf(X_g, Y_g, Z_g, levels=15, cmap='GnBu')
ax2d.scatter(hx, hy, c='red', s=30)
ax2d.scatter([gx], [gy], c='gold', s=100, marker='*')
ax2d.set_title(f"Iteration: {frame_idx} | Best Z: {gz:.4f}")
ax2d.set_xlim(config['x_bound'])
ax2d.set_ylim(config['y_bound'])

st.pyplot(fig)