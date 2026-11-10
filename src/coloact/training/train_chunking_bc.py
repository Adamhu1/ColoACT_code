import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 使用你的 A100
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from coloact.data.dataset import ColonDatasetDiffusion # 复用现有的 Dataset
from coloact.models.chunking_bc import ColonBCPolicy # 导入上面的 BC 模型
import wandb
import torch.nn as nn

# === 配置 (保持与 Diffusion 一致以便对比) ===
EXPERIMENT_MODE = "rgbd_elev" # 也可以测 rgb 或 rgbd
CHANNEL_MAP = {"rgb": 3, "rgbd": 4, "rgbd_elev": 5}

CONFIG = {
    "batch_size": 64, # BC 计算量小，Batch 可以很大
    "learning_rate": 1e-3, # BC 通常可以用大一点的学习率
    "epochs": 100,
    "pred_horizon": 16,
    "data_dir": ".data/h5", 
    "save_dir": f"./checkpoints_bc_{EXPERIMENT_MODE}", 
    "input_mode": EXPERIMENT_MODE,
    "input_channels": CHANNEL_MAP[EXPERIMENT_MODE],
    "device": "cuda"
}

def train():
    wandb.init(project="colon-bc-baseline", name=f"bc_{EXPERIMENT_MODE}", config=CONFIG)
    config = wandb.config
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 1. 复用 Dataset (完全不用改)
    dataset = ColonDatasetDiffusion(
        config.data_dir, 
        pred_horizon=config.pred_horizon,
        input_mode=config.input_mode
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, 
        num_workers=32, pin_memory=True, persistent_workers=True
    )
    
    # 2. 初始化 BC 模型
    model = ColonBCPolicy(
        pred_horizon=config.pred_horizon, 
        input_channels=config.input_channels
    ).to(config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss() # BC 使用均方误差

    print("🚀 开始训练 BC Baseline...")

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for step, (imgs, acts) in enumerate(train_loader):
            imgs = imgs.to(config.device)
            acts = acts.to(config.device) # GT: [B, 16, 2]
            
            optimizer.zero_grad()
            
            # Forward
            pred_acts = model(imgs) # Pred: [B, 16, 2]
            
            # Loss: 直接让预测动作逼近真实动作
            loss = criterion(pred_acts, acts)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        
        if (epoch+1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_dir, "bc_final.pth"))

    torch.save(model.state_dict(), os.path.join(config.save_dir, "bc_final.pth"))
    wandb.finish()

if __name__ == '__main__':
    train()