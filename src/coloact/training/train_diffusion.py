import os
# 指定使用 1 号显卡 (对应你的 A100)
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from coloact.data.dataset import ColonDatasetDiffusion
from coloact.models.diffusion import ColonDiffusionPolicy
import wandb

# =========================================================
# 🔥 实验控制中心 (在这里修改来做 Ablation Study)
# =========================================================
# 可选项: "rgb", "rgbd", "rgbd_elev"
EXPERIMENT_MODE = "rgbd_elev" 
# EXPERIMENT_MODE = "rgbd" 
# EXPERIMENT_MODE = "rgb" 
# =========================================================

# 自动映射通道数
CHANNEL_MAP = {
    "rgb": 3,
    "rgbd": 4,
    "rgbd_elev": 5
}

CONFIG = {
    "batch_size": 64,   # A100 优化值
    "num_workers": 32,    # A100 优化值
    "learning_rate": 1e-4,
    "epochs": 100,
    "pred_horizon": 16,
    "data_dir": "data/h5", # 👈 确保这里指向你真实的 H5 文件夹
    "save_dir": f"./checkpoints6_{EXPERIMENT_MODE}", # 自动存到不同文件夹
    "weight_decay": 1e-6,
    "input_mode": EXPERIMENT_MODE,
    "input_channels": CHANNEL_MAP[EXPERIMENT_MODE],
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def train():
    wandb.init(
        project="colon-diffusion-ablations", 
        name=f"run_{EXPERIMENT_MODE}_unconditional",
        config=CONFIG
    )
    config = wandb.config

    os.makedirs(config.save_dir, exist_ok=True)
    
    # 1. Dataset 初始化 (传入 input_mode)
    dataset = ColonDatasetDiffusion(
        config.data_dir, 
        pred_horizon=config.pred_horizon,
        input_mode=config.input_mode # <--- 关键参数
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    # 2. DataLoader (A100 性能优化)
    train_loader = DataLoader(
        train_data, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=True,
        persistent_workers=True # 保持子进程活跃
    )
    
    # 3. Model 初始化 (传入 input_channels, 去除 num_skills)
    model = ColonDiffusionPolicy(
        pred_horizon=config.pred_horizon, 
        input_channels=config.input_channels # <--- 关键参数
    ).to(config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    print(f"🚀 开始训练: {EXPERIMENT_MODE} | Channels: {config.input_channels} | Batch: {config.batch_size}")
    
    global_step = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        # 4. 训练循环 (只解包 imgs 和 acts，不再有 skills)
        for step, (imgs, acts) in enumerate(train_loader):
            imgs = imgs.to(config.device)
            acts = acts.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward
            loss = model(imgs, action_gt=acts)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            global_step += 1
            
            wandb.log({
                "train_batch_loss": current_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
                "global_step": global_step
            })
            
        lr_scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.epochs} | Avg Loss: {avg_loss:.6f}")
        
        wandb.log({"train_epoch_loss": avg_loss, "epoch": epoch})
        
        # 每 20 Epoch 保存一次
        if (epoch+1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(config.save_dir, f"ckpt_{epoch+1}.pth"))
            
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(config.save_dir, "diffusion_final.pth"))
    print(f"✅ 训练完成: {EXPERIMENT_MODE}")
    wandb.finish()

if __name__ == '__main__':
    train()