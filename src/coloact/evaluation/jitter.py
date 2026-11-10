import torch
import torchvision.transforms as transforms
import numpy as np
import h5py
import os
import glob
import random
from tqdm import tqdm

# --- 引入你的模型 ---
from coloact.models.act import ACTPolicy 

# ================= ⚙️ 配置区域 =================
CONFIG = {
    # 数据集目录
    "data_dir": "data/h5", 
    
    "model_path": "outputs/Checkpoints_ACT_RGBDE/act_best.pth",
    "input_mode": "rgbd_elev",
    "pred_horizon": 16,
    "dt": 0.1,
    "img_size": (240, 320), 
    "ACTION_MAX": np.array([3.0, 3.0]), 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 🔥 随机抽取多少个样本进行统计？(太多会跑很久，20-50个足够了)
    "num_samples_to_test": 50 
}
# =================================================

def unnormalize_action(action_norm, max_val):
    return action_norm * max_val

def calculate_jitter(action_seq):
    """
    计算抖动得分 (动作的一阶差分绝对值之和)
    Input: (T, 2)
    Output: float (平均每步的变化量)
    """
    # 只需要计算角速度 (Index 1) 的抖动，因为线速度通常比较稳
    w = action_seq[:, 1]
    
    # 差分: w[t+1] - w[t]
    diff = np.abs(np.diff(w))
    
    # 求平均
    return np.mean(diff)

def simulate_inference_for_stats(model, h5_path, device):
    """
    逻辑和画图代码一样，但不画图，只返回动作序列
    """
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(CONFIG["img_size"]),
        transforms.ToTensor()
    ])
    
    try:
        with h5py.File(h5_path, 'r') as root:
            rgbs = root['rgb'][:]          
            depths = root['depth'][:]      
            elevs = root['elevation'][:]   
    except Exception as e:
        print(f"❌ 读取错误 {h5_path}: {e}")
        return None, None

    T = rgbs.shape[0]
    # 如果轨迹太短，跳过
    if T < CONFIG['pred_horizon'] + 5:
        return None, None
        
    raw_actions_k1 = []   
    action_buffer = [[] for _ in range(T + CONFIG['pred_horizon'])]
    
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
            
            # k=1 Logic
            raw_actions_k1.append(pred_np[0]) 
            
            # k=16 Logic
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

def main():
    # 1. 加载模型
    channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
    model = ACTPolicy(
        pred_horizon=CONFIG["pred_horizon"], 
        input_channels=channel_map[CONFIG["input_mode"]]
    ).to(CONFIG["device"])
    
    state_dict = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # 2. 获取所有 h5 文件
    all_files = glob.glob(os.path.join(CONFIG["data_dir"], "*.h5"))
    if len(all_files) == 0:
        print("❌ 没有找到 .h5 文件")
        return

    # 3. 随机抽取
    num_samples = min(CONFIG["num_samples_to_test"], len(all_files))
    test_files = random.sample(all_files, num_samples)
    
    print(f"📊 开始统计实验: 在 {num_samples} 个随机样本上对比 k=1 vs k=16...")
    
    jitter_k1_list = []
    jitter_k16_list = []
    
    # 4. 循环测试
    for h5_path in tqdm(test_files):
        raw, ens = simulate_inference_for_stats(model, h5_path, CONFIG["device"])
        
        if raw is None: continue
        
        # 反归一化
        raw_phys = unnormalize_action(raw, CONFIG["ACTION_MAX"])
        ens_phys = unnormalize_action(ens, CONFIG["ACTION_MAX"])
        
        # 计算抖动 (Jitter)
        j1 = calculate_jitter(raw_phys)
        j16 = calculate_jitter(ens_phys)
        
        jitter_k1_list.append(j1)
        jitter_k16_list.append(j16)
        
    # 5. 统计结果
    mean_j1 = np.mean(jitter_k1_list)
    std_j1 = np.std(jitter_k1_list)
    
    mean_j16 = np.mean(jitter_k16_list)
    std_j16 = np.std(jitter_k16_list)
    
    reduction = (mean_j1 - mean_j16) / mean_j1 * 100
    
    print("\n" + "="*40)
    print("📈 Temporal Ensembling 统计验证结果")
    print("="*40)
    print(f"样本数量: {len(jitter_k1_list)}")
    print("-" * 30)
    print(f"🔴 无集成 (k=1)  Jitter: {mean_j1:.4f} ± {std_j1:.4f}")
    print(f"🔵 有集成 (k=16) Jitter: {mean_j16:.4f} ± {std_j16:.4f}")
    print("-" * 30)
    print(f"🚀 抖动抑制率 (Reduction): {reduction:.2f}%")
    print("="*40)
    
    # 简单的判定
    if mean_j16 < mean_j1:
        print("✅ 结论验证成功：Temporal Ensembling 显著降低了控制抖动。")
    else:
        print("❌ 结论验证失败：未发现明显改善。")

if __name__ == "__main__":
    main()