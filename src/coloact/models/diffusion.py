import torch
import torch.nn as nn
import torchvision.models as models
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# === 1. 1D Conditional U-Net (保持不变) ===
class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, local_cond_dim):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv1d(input_dim + local_cond_dim, 256, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.Mish()
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.Mish()
        )
        self.up1 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.Mish()
        )
        self.final_conv = nn.Conv1d(256, input_dim, kernel_size=3, padding=1)

    def forward(self, sample, local_cond):
        x = torch.cat([sample, local_cond], dim=1)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x_up = self.up1(x2)
        x_up = x_up + x1
        out = self.final_conv(x_up)
        return out

# === 2. 主策略类 (去除了 Skill，支持动态通道) ===
class ColonDiffusionPolicy(nn.Module):
    def __init__(self, action_dim=2, pred_horizon=16, vision_feature_dim=512, input_channels=5):
        """
        input_channels: 3(RGB) / 4(RGBD) / 5(RGBD+Elev)
        """
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        
        # === Vision Encoder ===
        self.vision_encoder = models.resnet18(weights=None)
        
        # ✅ 动态修改第一层卷积，适应不同的输入通道数
        self.vision_encoder.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.vision_encoder.fc = nn.Identity()
        
        # ❌ [已删除] Skill Embedding
        
        # === Noise Predictor ===
        # 条件维度 = 视觉(512) + 时间(1) [不再包含 Skill(64)]
        cond_dim = vision_feature_dim + 1 
        
        self.noise_net = ConditionalUnet1D(input_dim=action_dim, local_cond_dim=cond_dim)
        
        # === Noise Scheduler ===
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def forward(self, image, action_gt=None):
        """
        训练 Loop - 不再接收 skill 参数
        """
        device = image.device
        batch_size = image.shape[0]
        
        # 1. 提取视觉特征
        vision_feat = self.vision_encoder(image) # [B, 512]
        
        # 2. 采样噪声和时间步
        noise = torch.randn(action_gt.shape, device=device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        
        # 3. 加噪
        noisy_actions = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        noisy_actions_in = noisy_actions.permute(0, 2, 1) # [B, 16, 2] -> [B, 2, 16]
        
        # 4. 构造条件 (Vision + Time)
        time_emb = timesteps.unsqueeze(1).float() 
        
        # ✅ 只拼接 Vision 和 Time
        global_cond = torch.cat([vision_feat, time_emb], dim=1) # [B, 513]
        global_cond = global_cond.unsqueeze(2).expand(-1, -1, self.pred_horizon)
        
        # 5. 预测噪声
        noise_pred = self.noise_net(noisy_actions_in, global_cond)
        noise_pred = noise_pred.permute(0, 2, 1) # [B, 2, 16] -> [B, 16, 2]
        
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def predict_action(self, image):
        """
        推理 Loop - 不再接收 skill 参数
        """
        device = image.device
        batch_size = image.shape[0]
        
        vision_feat = self.vision_encoder(image)
        
        noisy_action = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim), 
            device=device
        )
        
        self.noise_scheduler.set_timesteps(100)
        
        for t in self.noise_scheduler.timesteps:
            time_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            time_emb = time_tensor.unsqueeze(1).float()
            
            # ✅ 只拼接 Vision 和 Time
            global_cond = torch.cat([vision_feat, time_emb], dim=1)
            global_cond = global_cond.unsqueeze(2).expand(-1, -1, self.pred_horizon)
            
            noisy_action_in = noisy_action.permute(0, 2, 1)
            noise_pred = self.noise_net(noisy_action_in, global_cond)
            noise_pred = noise_pred.permute(0, 2, 1)
            
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, 
                timestep=t, 
                sample=noisy_action
            ).prev_sample
            
        return noisy_action