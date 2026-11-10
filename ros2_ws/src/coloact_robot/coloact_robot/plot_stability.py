# 文件名: plot_scientific.py
# 这是一个用于生成科研级(Scientific-grade)稳定性分析图的完整脚本

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

# 设置 matplotlib 字体，避免保存时字体缺失警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

def plot_scientific_stability(h5_path, start_frame, num_frames=5, frame_interval=2, scanline_y=120):
    """
    绘制符合学术标准的稳定性分析图 (Global Normalization + Quantitative Metrics)
    """
    
    # --- 1. 数据加载与预处理 ---
    if not os.path.exists(h5_path):
        print(f"❌ 错误: 找不到文件: {h5_path}")
        return

    try:
        f = h5py.File(h5_path, 'r')
        total_len = f['depth'].shape[0]
        
        # 索引检查
        if start_frame + (num_frames-1)*frame_interval >= total_len:
            print(f"❌ 错误: 请求的帧范围超出文件长度 ({total_len})")
            return

        print(f"📄 正在处理: {os.path.basename(h5_path)}")
        
        indices = [start_frame + i * frame_interval for i in range(num_frames)]
        
        # 读取数据 (N, H, W, 1) -> (N, H, W)
        # 注意：这里假设 h5 数据是 float 类型。如果是 int (mm单位)，可能需要 /1000.0 转米
        raw_depths = f['depth'][indices][:, :, :, 0]
        raw_elevs = f['elevation'][indices][:, :, :, 0]
        
        f.close()
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # --- 2. 关键步骤：计算全局极值 (Global Min/Max) ---
    # 这是科学性的核心：必须用同一把“尺子”去衡量所有帧
    # 排除 inf 或 nan 的影响
    valid_depth_mask = np.isfinite(raw_depths)
    if not valid_depth_mask.any():
        print("❌ 数据全为 NaN 或 Inf")
        return
        
    d_global_min = raw_depths[valid_depth_mask].min()
    d_global_max = raw_depths[valid_depth_mask].max()
    d_range = d_global_max - d_global_min + 1e-6

    e_global_min = raw_elevs.min()
    e_global_max = raw_elevs.max()
    e_range = e_global_max - e_global_min + 1e-6

    print(f"📊 深度范围: [{d_global_min:.2f}, {d_global_max:.2f}]")
    print(f"📊 高程范围: [{e_global_min:.2f}, {e_global_max:.2f}]")

    # --- 3. 绘图布局初始化 ---
    fig = plt.figure(figsize=(16, 10), dpi=150)
    # 布局：前两行画图 (Depth, Elev)，第三行画大曲线
    gs = fig.add_gridspec(3, num_frames + 2, height_ratios=[1, 1, 1.2]) 

    # 准备色卡
    # 深度图用 Magma (感知均匀，黑-紫-亮黄)，高程图用 Terrain 或 Bone
    cmap_depth = 'magma'
    cmap_elev = 'bone'

    # --- 4. 循环绘制图像序列 ---
    for i in range(num_frames):
        # --- Depth Map ---
        ax_d = fig.add_subplot(gs[0, i])
        
        # 全局归一化用于显示
        d_vis = (raw_depths[i] - d_global_min) / d_range
        d_vis = np.clip(d_vis, 0, 1) # 确保不越界
        
        # 画扫描线 (白色虚线，增加可读性)
        # 注意：在 float 图像上画线，颜色要是 1.0 (白)
        d_vis_line = d_vis.copy()
        cv2.line(d_vis_line, (0, scanline_y), (d_vis.shape[1], scanline_y), 1.0, 1)
        
        ax_d.imshow(d_vis_line, cmap=cmap_depth, vmin=0, vmax=1)
        ax_d.axis('off')
        ax_d.set_title(f'T + {i*frame_interval*0.1:.1f}s', fontsize=11)
        
        # 第一列添加行标题
        if i == 0:
            ax_d.text(-0.2, 0.5, 'Relative\nDepth', transform=ax_d.transAxes, 
                     fontsize=14, rotation=90, va='center', fontweight='bold')

        # --- Elevation Map ---
        ax_e = fig.add_subplot(gs[1, i])
        
        # 全局归一化
        e_vis = (raw_elevs[i] - e_global_min) / e_range
        
        ax_e.imshow(e_vis, cmap=cmap_elev, vmin=0, vmax=1)
        ax_e.axis('off')
        
        if i == 0:
            ax_e.text(-0.2, 0.5, 'Pseudo\nElevation', transform=ax_e.transAxes, 
                     fontsize=14, rotation=90, va='center', fontweight='bold')

    # --- 5. 绘制科学分析曲线 (Scientific Profile Plot) ---
    ax_plot = fig.add_subplot(gs[2, :]) # 占据整行
    
    # 提取所有帧的扫描线数据 (Num_Frames, Width)
    scanlines = raw_depths[:, scanline_y, :]
    
    # 统一归一化 (关键！)
    scanlines_norm = (scanlines - d_global_min) / d_range
    
    # 计算统计指标
    mean_line = np.mean(scanlines_norm, axis=0)
    std_line = np.std(scanlines_norm, axis=0)
    
    # 计算整体平均标准差 (单一数值指标)
    avg_std_metric = np.mean(std_line)
    
    x_axis = np.arange(scanlines.shape[1])

    # 5.1 绘制单帧细线 (背景，低透明度)
    colors = plt.cm.viridis(np.linspace(0, 1, num_frames))
    for i in range(num_frames):
        ax_plot.plot(x_axis, scanlines_norm[i], color=colors[i], alpha=0.3, linewidth=1)

    # 5.2 绘制 Mean 线 (前景，醒目)
    ax_plot.plot(x_axis, mean_line, color='#D32F2F', linewidth=2.5, label='Mean Profile')

    # 5.3 绘制 Std Dev 阴影 (Uncertainty Band)
    ax_plot.fill_between(x_axis, 
                         mean_line - std_line, 
                         mean_line + std_line, 
                         color='#D32F2F', alpha=0.2, 
                         label=f'Std Dev (Jitter Impact)')

    # 图表装饰
    ax_plot.set_title(f"Temporal Stability Analysis at Scanline Y={scanline_y}\n"
                      f"Metric: Mean Std Dev = {avg_std_metric:.4f} (Lower is Better)", 
                      fontsize=14, fontweight='bold')
    
    ax_plot.set_xlabel("Image Width (Pixel)", fontsize=12)
    ax_plot.set_ylabel("Normalized Depth (Global Scale)", fontsize=12)
    ax_plot.set_ylim(-0.05, 1.05)
    ax_plot.grid(True, linestyle='--', alpha=0.4)
    ax_plot.legend(loc='upper right', frameon=True, fontsize=10)
    
    # --- 6. 保存与显示 ---
    plt.tight_layout()
    
    # 自动生成文件名，包含指标
    save_filename = f"stability_metric_{avg_std_metric:.4f}.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ 图表生成完毕！")
    print(f"📊 稳定性量化指标 (Mean Std): {avg_std_metric:.4f}")
    print(f"💾 已保存为: {os.path.abspath(save_filename)}")
    
    plt.show()

# --- 主程序入口 ---
if __name__ == "__main__":
    TARGET_FILE = "/path/to/ros2_ws/src/coloact_robot/dataset_labeled/skill2_20251214_170054.h5"
    # 2. 检查文件是否存在
    if os.path.exists(TARGET_FILE):
        h5_full_path = TARGET_FILE
    else:
        print(f"❌ 错误：找不到指定的文件: {TARGET_FILE}")
        print("请检查路径是否正确，或者是否输错了文件名。")
        sys.exit(1)

    # 3. 运行配置
    # START_FRAME: 请务必在播放器里确认这一秒是否真的发生了抖动
    # 比如第 5 秒开始抖，如果是 10fps，就是第 50 帧
    START_FRAME = 50 
    
    print(f"🚀 开始分析文件: {os.path.basename(h5_full_path)}")
    
    plot_scientific_stability(
        h5_path=h5_full_path, 
        start_frame=START_FRAME, 
        num_frames=5,        # 展示 5 帧
        frame_interval=2,    # 每隔 2 帧取样一次
        scanline_y=120       # 扫描线高度 (0-240)
    )