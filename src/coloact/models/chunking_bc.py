import torch
import torch.nn as nn
import torchvision.models as models

class ColonBCPolicy(nn.Module):
    def __init__(self, action_dim=2, pred_horizon=16, input_channels=5):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        
        # === 1. 视觉编码器 (必须与 Diffusion 保持完全一致) ===
        # 这样才能证明是 Diffusion 算法强，而不是 ResNet 强
        self.vision_encoder = models.resnet18(weights=None)
        
        # 修改第一层卷积以适应 3/4/5 通道
        self.vision_encoder.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        # ResNet18 fc 输出是 1000，我们改成 Identity，取 avgpool 之前的特征
        # 或者直接用 fc 输出 512
        self.vision_encoder.fc = nn.Identity() 
        
        # ResNet18 最后一层 avgpool 出来的特征数是 512
        vision_feature_dim = 512
        
        # === 2. 策略头 (简单的 MLP) ===
        # 输入: 视觉特征 (512)
        # 输出: 动作序列 (16 * 2 = 32)
        self.policy_head = nn.Sequential(
            nn.Linear(vision_feature_dim, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, pred_horizon * action_dim) # 直接输出所有步数的动作
        )

    def forward(self, image):
        """
        训练/推理 通用
        Output: [Batch, 16, 2]
        """
        # 1. 提取特征
        vision_feat = self.vision_encoder(image) # [B, 512]
        
        # 2. 回归预测
        action_flat = self.policy_head(vision_feat) # [B, 32]
        
        # 3. Reshape 成序列 [B, 16, 2]
        action_pred = action_flat.view(-1, self.pred_horizon, self.action_dim)
        
        return action_pred