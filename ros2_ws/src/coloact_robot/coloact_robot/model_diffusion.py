import torch
import torch.nn as nn
import torchvision.models as models
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# === 1. 1D Conditional U-Net (淇濇寔涓嶅彉) ===
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

# === 2. 涓荤瓥鐣ョ被 (鍘婚櫎浜?Skill锛屾敮鎸佸姩鎬侀€氶亾) ===
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
        
        # 鉁?鍔ㄦ€佷慨鏀圭涓€灞傚嵎绉紝閫傚簲涓嶅悓鐨勮緭鍏ラ€氶亾鏁?
        self.vision_encoder.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.vision_encoder.fc = nn.Identity()
        
        # 鉂?[宸插垹闄 Skill Embedding
        
        # === Noise Predictor ===
        # 鏉′欢缁村害 = 瑙嗚(512) + 鏃堕棿(1) [涓嶅啀鍖呭惈 Skill(64)]
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
        璁粌 Loop - 涓嶅啀鎺ユ敹 skill 鍙傛暟
        """
        device = image.device
        batch_size = image.shape[0]
        
        # 1. 鎻愬彇瑙嗚鐗瑰緛
        vision_feat = self.vision_encoder(image) # [B, 512]
        
        # 2. 閲囨牱鍣０鍜屾椂闂存
        noise = torch.randn(action_gt.shape, device=device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        
        # 3. 鍔犲櫔
        noisy_actions = self.noise_scheduler.add_noise(action_gt, noise, timesteps)
        noisy_actions_in = noisy_actions.permute(0, 2, 1) # [B, 16, 2] -> [B, 2, 16]
        
        # 4. 鏋勯€犳潯浠?(Vision + Time)
        time_emb = timesteps.unsqueeze(1).float() 
        
        # 鉁?鍙嫾鎺?Vision 鍜?Time
        global_cond = torch.cat([vision_feat, time_emb], dim=1) # [B, 513]
        global_cond = global_cond.unsqueeze(2).expand(-1, -1, self.pred_horizon)
        
        # 5. 棰勬祴鍣０
        noise_pred = self.noise_net(noisy_actions_in, global_cond)
        noise_pred = noise_pred.permute(0, 2, 1) # [B, 2, 16] -> [B, 16, 2]
        
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def predict_action(self, image):
        """
        鎺ㄧ悊 Loop - 涓嶅啀鎺ユ敹 skill 鍙傛暟
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
            
            # 鉁?鍙嫾鎺?Vision 鍜?Time
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


