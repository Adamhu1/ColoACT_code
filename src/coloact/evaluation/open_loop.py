import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import glob

# --- 引入自定义模块 ---
from coloact.data.dataset import ColonDatasetDiffusion
from coloact.models.act import ACTPolicy 

# ================= ⚙️ 配置区域 (CONFIG) =================
CONFIG = {
    # 1. 路径
    "data_dir": "data/h5",
    "model_path": "outputs/Checkpoints_ACT_RGBDE/act_best.pth",
    "input_mode": "rgbd_elev",
    "pred_horizon": 16,
    
    # 2. 物理参数
    "dt": 1.0 / 15.0,     # 15Hz ≈ 0.067s
    
    # 3. 缩放与限制
    "SCALE_PRED": np.array([3.0, 3.0]),  # 把归一化的输出还原为物理单位
    "LIMIT_V": [-0.5, 0.5],              # 线速度物理限制 (m/s)
    "LIMIT_W": [-1.5, 1.5],              # 角速度物理限制 (rad/s)
    
    # 4. 绘图设置
    "step_size": 5,       # 每隔 10 帧画一次 (解决太密看不清的问题)
    "plot_duration": 400,  # 画前 400 帧
    
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
# =========================================================

def integrate_chunk(actions, init_pose=(0,0,0), dt=0.1):
    x, y, theta = init_pose
    traj = [(x, y)]
    for v, w in actions:
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        traj.append((x, y))
    return np.array(traj)

def visualize_global_dense_marked(model, h5_file_path):
    print(f"\n🎨 正在绘制带标记的全局轨迹: {os.path.basename(h5_file_path)}")
    
    with h5py.File(h5_file_path, 'r') as f:
        full_action = f['action'][:] 
        full_rgb = f['rgb'][:]
        full_depth = f['depth'][:]
        full_elev = f['elevation'][:]
    
    max_steps = min(len(full_action) - CONFIG["pred_horizon"], CONFIG["plot_duration"])
    
    # --- 1. 积分 Ground Truth (灰色长龙) ---
    poses = [(0, 0, 0)] 
    x, y, theta = 0, 0, 0
    gt_traj = [(0, 0)]
    
    # 注意：H5里的action已经是物理单位，不需要缩放
    for v, w in full_action[:max_steps]:
        x += v * np.cos(theta) * CONFIG["dt"]
        y += v * np.sin(theta) * CONFIG["dt"]
        theta += w * CONFIG["dt"]
        gt_traj.append((x, y))
        poses.append((x, y, theta))
    gt_traj = np.array(gt_traj)

    # --- 2. 开始画图 ---
    plt.figure(figsize=(12, 10))
    
    # 画 GT (灰色粗线)
    plt.plot(gt_traj[:, 1], gt_traj[:, 0], 'k-', linewidth=4, alpha=0.3, label='Expert Trajectory')
    
    # --- 3. 密集预测 ---
    print("   计算 Predictions...")
    model.eval()
    
    # 用于图例去重
    has_labeled_pred = False
    has_labeled_point = False
    
    with torch.no_grad():
        for t in range(0, max_steps, CONFIG["step_size"]):
            # (A) 数据预处理
            img_rgb = full_rgb[t]
            img_depth = full_depth[t]
            img_elev = full_elev[t]
            
            # 转 Tensor
            rgb_t = torch.from_numpy(img_rgb).float() / 255.0
            if rgb_t.ndim == 3 and rgb_t.shape[-1] == 3: rgb_t = rgb_t.permute(2, 0, 1)
            
            depth_t = torch.from_numpy(img_depth).float()
            if depth_t.ndim == 2: depth_t = depth_t.unsqueeze(0)
            elif depth_t.ndim == 3 and depth_t.shape[-1] == 1: depth_t = depth_t.permute(2, 0, 1)
                
            elev_t = torch.from_numpy(img_elev).float() / 255.0
            if elev_t.ndim == 2: elev_t = elev_t.unsqueeze(0)
            elif elev_t.ndim == 3 and elev_t.shape[-1] == 1: elev_t = elev_t.permute(2, 0, 1)
            
            obs = torch.cat([rgb_t, depth_t, elev_t], dim=0).unsqueeze(0).to(CONFIG["device"])
            
            # (B) 推理
            pred_actions = model(obs, actions=None)[0]
            pred_norm = pred_actions.cpu().numpy()[0]
            
            # (C) 物理还原 + 限制 (关键修正)
            raw_phys = pred_norm * CONFIG["SCALE_PRED"]
            pred_phys = np.zeros_like(raw_phys)
            pred_phys[:, 0] = np.clip(raw_phys[:, 0], CONFIG["LIMIT_V"][0], CONFIG["LIMIT_V"][1])
            pred_phys[:, 1] = np.clip(raw_phys[:, 1], CONFIG["LIMIT_W"][0], CONFIG["LIMIT_W"][1])
            
            # (D) 积分
            curr_pose = poses[t] 
            chunk_traj = integrate_chunk(pred_phys, init_pose=curr_pose, dt=CONFIG["dt"])
            
            # (E) 画图核心修改
            # 1. 画预测线 (红色)
            label_line = "ACT Prediction (16-steps)" if not has_labeled_pred else ""
            plt.plot(chunk_traj[:, 1], chunk_traj[:, 0], 'r-', linewidth=1.0, alpha=0.5, label=label_line)
            if not has_labeled_pred: has_labeled_pred = True
            
            # 2. 画规划起点 (蓝色圆点) —— 这就是您要的“切点”
            label_pt = "Planning Point" if not has_labeled_point else ""
            # chunk_traj[0] 就是起点 (t时刻的真实位置)
            plt.scatter(chunk_traj[0, 1], chunk_traj[0, 0], c='blue', s=15, zorder=5, alpha=0.9, label=label_pt)
            if not has_labeled_point: has_labeled_point = True

    # 装饰
    plt.title(f"Global Rollout with Planning Points: {os.path.basename(h5_file_path)}", fontsize=14)
    plt.xlabel("Lateral (m)")
    plt.ylabel("Forward (m)")
    plt.legend(loc='best')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_name = f"vis_marked_{os.path.basename(h5_file_path)[:-3]}.png"
    plt.savefig(save_name)
    print(f"✅ 完成！图片已保存: {save_name}")

if __name__ == "__main__":
    # 初始化模型
    channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
    model = ACTPolicy(
        pred_horizon=CONFIG["pred_horizon"], 
        input_channels=channel_map[CONFIG["input_mode"]]
    ).to(CONFIG["device"])
    
    state_dict = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    # 1. 把文件名填在这里 (确保这个文件在您的 data_dir 目录下)
    target_filename = "skill0_20251226_162528.h5"  # <--- 修改成您想画的那个文件名
    
    # 2. 拼接完整路径
    target_file_path = os.path.join(CONFIG["data_dir"], target_filename)
    
    # 3. 检查文件是否存在并运行
    if os.path.exists(target_file_path):
        visualize_global_dense_marked(model, target_file_path)
    else:
        print(f"❌ 找不到文件: {target_file_path}")
        print(f"请检查文件名是否正确，或者路径: {CONFIG['data_dir']} 是否正确")