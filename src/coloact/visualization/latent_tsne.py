import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import os

# --- 引入你的模块 ---
from coloact.data.dataset import ColonDatasetDiffusion
from coloact.models.act import ACTPolicy

# ================= ⚙️ 配置区域 =================
CONFIG = {
    "data_dir": "data/h5",
    "model_path": "outputs/Checkpoints_ACT_RGBDE/act_best.pth",
    "input_mode": "rgbd_elev",
    "pred_horizon": 16,
    "batch_size": 32, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 即使画连续图，我们还是保留离散标签在CSV里，方便你以后在Origin里备用
    "TURN_THRESHOLD": 0.15, 
    
    # t-SNE 参数
    "PERPLEXITY": 30, 
    "N_ITER": 1000,
    
    "export_file": "tsne_latent_data.csv"
}
# =================================================

def classify_action(actions_batch):
    """
    保留此函数用于生成 CSV 标签
    """
    avg_w = torch.mean(actions_batch[:, :, 1], dim=1).cpu().numpy() 
    labels = []
    for w in avg_w:
        if w > CONFIG["TURN_THRESHOLD"]:
            labels.append("Left")
        elif w < -CONFIG["TURN_THRESHOLD"]:
            labels.append("Right")
        else:
            labels.append("Straight")
    return labels, avg_w

def extract_latent_space():
    print(f"🚀 正在准备数据...")
    
    full_dataset = ColonDatasetDiffusion(
        data_dir=CONFIG["data_dir"],
        pred_horizon=CONFIG["pred_horizon"],
        input_mode=CONFIG["input_mode"]
    )
    
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    print(f"📊 验证集样本数: {len(val_dataset)}")

    channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
    model = ACTPolicy(
        pred_horizon=CONFIG["pred_horizon"], 
        input_channels=channel_map[CONFIG["input_mode"]]
    ).to(CONFIG["device"])
    
    state_dict = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print("🔄 开始提取潜变量 (Latent z)...")
    
    latents = []
    all_labels = []
    all_avg_w = []
    
    with torch.no_grad():
        for obs, actions in val_loader:
            obs = obs.to(CONFIG["device"])
            actions = actions.to(CONFIG["device"])
            
            if isinstance(obs, (tuple, list)):
                _, mu, _ = model(obs[0], qpos=obs[1], actions=actions)
            else:
                _, mu, _ = model(obs, actions=actions)
            
            latents.append(mu.cpu().numpy())
            
            batch_labels, batch_w = classify_action(actions)
            all_labels.extend(batch_labels)
            all_avg_w.extend(batch_w)
            
    latents = np.concatenate(latents, axis=0) 
    print(f"✅ 提取完成! Latent Shape: {latents.shape}")
    
    return latents, all_labels, all_avg_w

def run_tsne_and_plot(latents, labels, avg_w):
    print("🧮 正在运行 t-SNE 降维 (这可能需要几秒钟)...")
    
    tsne = TSNE(
        n_components=2, 
        perplexity=CONFIG["PERPLEXITY"], 
        n_iter=CONFIG["N_ITER"], 
        random_state=42,
        init='pca', 
        learning_rate='auto'
    )
    z_embedded = tsne.fit_transform(latents)
    
    # --- 1. 导出数据给 Origin ---
    print(f"💾 正在导出 CSV: {CONFIG['export_file']} ...")
    df = pd.DataFrame({
        "tSNE_1": z_embedded[:, 0],
        "tSNE_2": z_embedded[:, 1],
        "Label": labels,             # 保留离散标签 (Left/Right)
        "Avg_Angular_Vel": avg_w     # 保留连续数值 (用于连续色谱)
    })
    df.to_csv(CONFIG['export_file'], index=False)
    
    # --- 2. Python 连续色谱预览 (Continuous Color Map) ---
    print("🎨 正在生成连续色谱预览图...")
    plt.figure(figsize=(11, 9)) # 稍微宽一点，给colorbar留位置
    
    # 🟢 核心修改：hue 绑定数值列，palette 使用 coolwarm
    scatter = sns.scatterplot(
        x="tSNE_1", y="tSNE_2",
        hue="Avg_Angular_Vel",   # 绑定到数值
        palette="coolwarm",      # 蓝色(负/右) -> 白色(0) -> 红色(正/左)
        data=df,
        alpha=0.8,               # 透明度
        s=60,
        edgecolor=None           # 去掉点的边框，看起来更顺滑
    )
    
    # 获取图例句柄 (Seaborn 对于数值hue会自动生成渐变图例)
    # 我们也可以手动设置 Colorbar 来让它更好看 (Optional, 但 Seaborn 默认的其实够用了)
    
    plt.title("Continuous Latent Space Visualization (Manifold Structure)", fontsize=15)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    save_name = "tsne_preview_continuous.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ 连续色谱预览图已保存: {save_name}")
    # plt.show() # 如果在服务器上，保持注释

if __name__ == "__main__":
    z_data, labels, w_data = extract_latent_space()
    run_tsne_and_plot(z_data, labels, w_data)