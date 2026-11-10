#  # 请确保这个 h5 文件存在，且包含 rgb, depth, elevation 键
#     "h5_path": "data/h5/skill0_20251219_141920.h5", 
    
#     "model_path": "outputs/Checkpoints_ACT_RGBDE2/act_best.pth",
#     "input_mode": "rgbd_elev",
#     "pred_horizon": 16,
#     "dt": 0.1,
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 🟢 新增：用于导出 CSV
import h5py
import os
from collections import deque

# --- 引入你的模型 ---
from coloact.models.act import ACTPolicy 

# ================= ⚙️ 配置区域 =================
CONFIG = {
    "h5_path": "data/h5/skill0_20251219_141920.h5", 
    "model_path": "outputs/Checkpoints_ACT_RGBDE2/act_best.pth",
    "input_mode": "rgbd_elev",
    "pred_horizon": 16,
    "dt": 0.1,
    "img_size": (240, 320), 
    "ACTION_MAX": np.array([3.0, 3.0]), 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 🟢 导出文件名
    "export_file": "temporal_ensembling_data.csv"
}
# =================================================

def unnormalize_action(action_norm, max_val):
    return action_norm * max_val

def simulate_inference(model, h5_path, device):
    print(f"📂 正在读取文件: {h5_path}")
    
    # 预处理
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(CONFIG["img_size"]),
        transforms.ToTensor()
    ])
    
    with h5py.File(h5_path, 'r') as root:
        rgbs = root['rgb'][:]          
        depths = root['depth'][:]      
        elevs = root['elevation'][:]   
        
    T = rgbs.shape[0]
    print(f"⏱️ 轨迹总长度: {T} 步")
    
    raw_actions_k1 = []   
    action_buffer = [[] for _ in range(T + CONFIG['pred_horizon'])]
    
    model.eval()
    
    print("🚀 开始逐帧推理...")
    with torch.no_grad():
        for t in range(T):
            img_np = rgbs[t] 
            depth_np = depths[t]
            if len(depth_np.shape) == 2: depth_np = np.expand_dims(depth_np, -1)
            elev_np = elevs[t]
            if len(elev_np.shape) == 2: elev_np = np.expand_dims(elev_np, -1)

            rgb_t = preprocess(img_np).to(device).unsqueeze(0)
            
            depth_t = torch.from_numpy(depth_np).permute(2,0,1).float().unsqueeze(0).to(device)
            depth_t = torch.nn.functional.interpolate(depth_t, size=CONFIG["img_size"], mode='bilinear')
            
            elev_t = torch.from_numpy(elev_np).permute(2,0,1).float().unsqueeze(0).to(device)
            elev_t = torch.nn.functional.interpolate(elev_t, size=CONFIG["img_size"], mode='bilinear')
            
            obs = torch.cat([rgb_t, depth_t, elev_t], dim=1) 
            
            pred_chunk = model(obs, actions=None)[0]
            pred_np = pred_chunk.cpu().numpy()[0]
            
            # 1. k=1 Raw
            raw_actions_k1.append(pred_np[0]) 
            
            # 2. k=16 Buffer
            for i in range(CONFIG['pred_horizon']):
                if t + i < len(action_buffer):
                    action_buffer[t + i].append(pred_np[i])

    ensembled_actions = []
    for t in range(T):
        preds = action_buffer[t]
        if len(preds) > 0:
            avg_action = np.mean(preds, axis=0) 
            ensembled_actions.append(avg_action)
        else:
            ensembled_actions.append(raw_actions_k1[t])

    return np.array(raw_actions_k1), np.array(ensembled_actions)

def export_and_plot(raw, ensembled):
    # ---------------------------------------------------------
    # 🔴 修改 1: 不再进行反归一化，直接使用模型输出的原始值
    # ---------------------------------------------------------
    # 假设模型的输出范围是 [-1, 1] 或者标准化后的分布
    # 我们直接取 index 1 (通常是转向/角速度分量)
    raw_w = raw[:, 1]       
    ens_w = ensembled[:, 1] 
    
    # 如果你的数据很小（比如在 0.01 级别），而且你想看清楚，
    # 你甚至可以乘 100 变成百分比，或者保持原样。
    # 这里保持原样：
    
    time = np.arange(len(raw_w)) * CONFIG["dt"]
    start, end = 0, len(time)
    
    # 2. 导出 CSV 给 Origin
    print(f"💾 正在导出数据到 {CONFIG['export_file']} ...")
    df = pd.DataFrame({
        "Time_s": time[start:end],
        "Raw_Action_k1": raw_w[start:end],        # 原始归一化值
        "Ensembled_Action_k16": ens_w[start:end]  # 原始归一化值
    })
    df.to_csv(CONFIG['export_file'], index=False)
    print("✅ CSV 导出成功！")

    # 3. 生成预览图
    plt.figure(figsize=(10, 6))
    
    plt.plot(df["Time_s"], df["Raw_Action_k1"], 
             color='gray', linestyle='--', alpha=0.6, label='No Ensemble (k=1)')
    plt.plot(df["Time_s"], df["Ensembled_Action_k16"], 
             color='#0072BD', linewidth=2.5, label='Ensembled (k=16)')
    
    plt.title("Temporal Ensembling Effect (Normalized Space)", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    
    # ---------------------------------------------------------
    # 🔴 修改 2: 修改 Y 轴标签，去掉物理单位
    # ---------------------------------------------------------
    plt.ylabel("Normalized Action Value", fontsize=12) 
    # 或者用 "Control Signal (-1 to 1)"
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    preview_filename = "preview_temporal_ensembling_norm.png"
    plt.savefig(preview_filename, dpi=150) 
    print(f"✅ 预览图已保存: {preview_filename}")

if __name__ == "__main__":
    channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
    model = ACTPolicy(
        pred_horizon=CONFIG["pred_horizon"], 
        input_channels=channel_map[CONFIG["input_mode"]]
    ).to(CONFIG["device"])
    
    state_dict = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    raw_act, ens_act = simulate_inference(model, CONFIG["h5_path"], CONFIG["device"])
    
    export_and_plot(raw_act, ens_act)