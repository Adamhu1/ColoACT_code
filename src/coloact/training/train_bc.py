import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from coloact.data.dataset import ColonDatasetDiffusion
from coloact.models.behavior_cloning import ColonBCSingleStep # 导入单步模型
import wandb

# === 配置 ===
EXPERIMENT_MODE = "rgbd_elev" 
CHANNEL_MAP = {"rgb": 3, "rgbd": 4, "rgbd_elev": 5}

CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-3, 
    "epochs": 100,
    "pred_horizon": 16, # Dataset 还是读 16 步，但我们只用第 1 步
    "data_dir": "data/h5", 
    "save_dir": f"./checkpoints_bc_singlestep0_{EXPERIMENT_MODE}", 
    "input_mode": EXPERIMENT_MODE,
    "input_channels": CHANNEL_MAP[EXPERIMENT_MODE],
    "device": "cuda"
}

def train():
    wandb.init(project="colon-bc-baseline", name=f"bc_singlestep_{EXPERIMENT_MODE}", config=CONFIG)
    config = wandb.config
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 1. 复用 Dataset
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
    
    # 2. 初始化单步模型
    model = ColonBCSingleStep(
        input_channels=config.input_channels
    ).to(config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    print("🚀 开始训练 Single-Step BC Baseline...")

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for step, (imgs, acts) in enumerate(train_loader):
            imgs = imgs.to(config.device)
            acts = acts.to(config.device) # GT: [B, 16, 2]
            
            optimizer.zero_grad()
            
            # Forward
            pred_curr_action = model(imgs) # Pred: [B, 2]
            
            # === 关键点: 只取 GT 的第 0 步进行对比 ===
            target_curr_action = acts[:, 0, :] # GT: [B, 2]
            
            loss = criterion(pred_curr_action, target_curr_action)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        
        if (epoch+1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_dir, "bc_singlestep_final.pth"))

    torch.save(model.state_dict(), os.path.join(config.save_dir, "bc_singlestep_final.pth"))
    wandb.finish()

if __name__ == '__main__':
    train()