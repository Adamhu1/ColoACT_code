import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# ================= ⚙️ 配置区域 =================
# 替换成你的 .h5 文件路径
FILE_PATH = "outputs/try_with_h5/skill0_20260106_164525.h5" 
DATA_KEY = 'action' 

# 假设 Index 0 = 线速度 (v), Index 1 = 角速度 (w)
# 如果图画出来横纵反了，请在这里互换
INDEX_V = 0
INDEX_W = 1
# =================================================

def plot_density():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 文件不存在: {FILE_PATH}")
        return

    print(f"📂 读取数据: {FILE_PATH} ...")
    try:
        with h5py.File(FILE_PATH, 'r') as f:
            actions = f[DATA_KEY][:]
            actions = np.array(actions)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 提取 v 和 w
    v_data = actions[:, INDEX_V]
    w_data = actions[:, INDEX_W]

    # 创建 DataFrame，方便 Seaborn 处理
    df = pd.DataFrame({
        "Angular Velocity (Turn)": w_data,
        "Linear Velocity (Forward)": v_data
    })

    # =========================================================
    # 🎨 核心绘图：使用 JointPlot (KDE 模式)
    # =========================================================
    print("🎨 正在生成密度热力图...")
    
    # kind="kde": 核密度估计 (画等高线/热力图)
    # fill=True: 填充颜色
    # cmap="rocket": 颜色主题 (深色代表高密度)
    # levels=10: 等高线层数
    g = sns.jointplot(
        data=df,
        x="Angular Velocity (Turn)",
        y="Linear Velocity (Forward)",
        kind="kde",   
        fill=True,
        cmap="Spectral_r",  # 推荐色谱：红色=高频，蓝色=低频
        thresh=0.05,        # 过滤掉密度极低的噪点
        levels=15,          # 层次更丰富
        height=8
    )

    # --- 添加辅助线 ---
    # 在主图(ax_joint)上画十字线
    g.ax_joint.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    g.ax_joint.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # --- 添加文字标注 (解释趋势) ---
    # 根据你的数据范围，手动调整这些文字的位置
    # 通常直行在高处中间，停止在 (0,0)，转弯在两侧
    g.ax_joint.text(0, np.max(v_data)*0.9, "Moving Straight", 
                    ha='center', color='black', fontweight='bold', fontsize=10)
    g.ax_joint.text(0, 0, "Stop / Hesitation", 
                    ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)
    
    # 调整坐标轴范围 (让图居中)
    max_w = max(abs(np.min(w_data)), abs(np.max(w_data))) * 1.1
    g.ax_joint.set_xlim(-max_w, max_w)
    
    # 设置标题 (稍微上移一点避免重叠)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Action Primitive Distribution (Density Analysis)", fontsize=16)

    save_name = "action_density_heatmap.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ 图表已保存: {save_name}")
    # plt.show() # 服务器上请注释掉

if __name__ == "__main__":
    plot_density()