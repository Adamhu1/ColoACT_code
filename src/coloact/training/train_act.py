import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from coloact.data.dataset import ColonDatasetDiffusion 
from coloact.models.act import ACTPolicy
import matplotlib.pyplot as plt
import os
# 🔥 【关键修改 1】 强制指定使用 2 号显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import wandb
import numpy as np

# 引入混合精度训练库 (加速训练)
from torch.cuda.amp import autocast, GradScaler 

# === 配置 ===
# 🔥 【关键修改 3】 修改保存路径，防止覆盖之前的模型
CONFIG = {
    "data_dir": "data/h5",
    "batch_size": 64,      # 如果显存够大，尽量大一点 (64 -> 128)
    "num_epochs": 100,
    "lr": 1e-4,
    "kl_weight": 10.0,      # ACT 论文建议权重
    "save_dir": "./Checkpoints_ACT_RGBD",
    "input_mode": "rgbd",   # 你的模式: rgbd 或 rgbd_elev
    "val_ratio": 0.2,       # 🔥 20% 的数据用于验证
    "num_workers": 8        # 数据加载线程数
}

def train():
    # 0. 基础设置
    # 开启 CUDNN 加速
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # 初始化 WandB
    wandb.init(
        # 🔥 【关键修改 4】 修改实验名称，方便在网页区分
        project="colon_robot_act_RGBD", 
        name=f"act_run_val_split_kl{CONFIG['kl_weight']}",
        config=CONFIG
    )

    # 1. 准备数据集 & 划分 训练/验证集
    full_dataset = ColonDatasetDiffusion(
        data_dir=CONFIG["data_dir"],
        pred_horizon=16,
        input_mode=CONFIG["input_mode"]
    )
    
    # 计算数量
    val_size = int(len(full_dataset) * CONFIG["val_ratio"])
    train_size = len(full_dataset) - val_size
    
    # 🔥 随机切分
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"📊 数据集详情: 总数 {len(full_dataset)} | 训练集 {train_size} | 验证集 {val_size}")

    # 创建 DataLoader
    # pin_memory=True 加速 CPU->GPU 传输
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )
    
    # 验证集不需要 shuffle
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )

    # 2. 模型与优化器
    channel_map = {'rgb': 3, 'rgbd': 4, 'rgbd_elev': 5}
    model = ACTPolicy(
        pred_horizon=16, 
        input_channels=channel_map[CONFIG["input_mode"]]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    
    # 初始化混合精度 Scaler
    scaler = GradScaler()

    # 记录最佳模型
    best_val_loss = float('inf')

    print("🚀 开始训练 ACT 模型 (带验证集评估)...")
    
    for epoch in range(CONFIG["num_epochs"]):
        # ==========================
        # 🟢 阶段 1: 训练 (Training)
        # ==========================
        model.train()
        train_loss_sum = 0
        train_mse_sum = 0
        train_kl_sum = 0
        
        for batch_idx, (obs, actions) in enumerate(train_loader):
            obs = obs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # 混合精度前向
            with autocast():
                pred_actions, mu, logvar = model(obs, actions)
                
                loss_mse = F.mse_loss(pred_actions, actions)
                loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss_kl = loss_kl / obs.shape[0] # batch mean
                
                loss = loss_mse + CONFIG["kl_weight"] * loss_kl

            # 混合精度反向
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_mse_sum += loss_mse.item()
            train_kl_sum += loss_kl.item()

        # 计算训练集平均 Loss
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_mse = train_mse_sum / len(train_loader)
        avg_train_kl = train_kl_sum / len(train_loader)

        # ==========================
        # 🔵 阶段 2: 验证 (Validation)
        # ==========================
        model.eval()
        val_loss_sum = 0
        val_mse_sum = 0
        
        # 🔥 禁止梯度计算 (节省显存，防止更新参数)
        with torch.no_grad():
            for obs, actions in val_loader:
                obs = obs.to(device, non_blocking=True)
                actions = actions.to(device, non_blocking=True)
                
                # Forward (不传 actions 也可以，但为了算 loss 我们传进去)
                # 注意：ACT在推理时不需要 actions，但在验证时为了算 loss 需要
                # 如果你的 model forward 写法里有 if actions is not None，这里会走 CVAE 路径
                # 严格来说验证集也应该看 CVAE 的重建能力
                with autocast():
                    pred_actions, mu, logvar = model(obs, actions)
                    
                    loss_mse = F.mse_loss(pred_actions, actions)
                    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss_kl = loss_kl / obs.shape[0]
                    
                    loss = loss_mse + CONFIG["kl_weight"] * loss_kl
                
                val_loss_sum += loss.item()
                val_mse_sum += loss_mse.item()
        
        # 计算验证集平均 Loss
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_mse = val_mse_sum / len(val_loader)

        # ==========================
        # 📝 日志与保存
        # ==========================
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} "
              f"| Train Loss: {avg_train_loss:.4f} (MSE: {avg_train_mse:.4f}) "
              f"| Val Loss: {avg_val_loss:.4f} (MSE: {avg_val_mse:.4f})")

        # WandB 记录两条曲线
        wandb.log({
            "epoch": epoch + 1,
            "loss/train": avg_train_loss,
            "loss/val": avg_val_loss,
            "mse/train": avg_train_mse,
            "mse/val": avg_val_mse,
            "kl/train": avg_train_kl,
            "lr": optimizer.param_groups[0]['lr']
        })

        # 🔥 策略 1: 保存最佳模型 (Best Checkpoint)
        # 如果当前验证集 Loss 是历史最低，就保存为 best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{CONFIG['save_dir']}/act_best.pth")
            print(f"🌟 新的最佳模型已保存 (Val Loss: {best_val_loss:.4f})")

        # 策略 2: 定期保存 (Regular Checkpoint)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{CONFIG['save_dir']}/act_epoch_{epoch+1}.pth")

    # 保存最终模型
    torch.save(model.state_dict(), f"{CONFIG['save_dir']}/act_final.pth")
    wandb.finish()
    print("✅ 训练全部完成！")

if __name__ == "__main__":
    train()