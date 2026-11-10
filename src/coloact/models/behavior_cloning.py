import torch
import torch.nn as nn
import torchvision.models as models

class ColonBCSingleStep(nn.Module):
    def __init__(self, action_dim=2, input_channels=5):
        super().__init__()
        
        # === 1. 视觉编码器 (保持与 Diffusion/Chunking BC 一致) ===
        # 必须用一样的骨干网络，确保公平
        self.vision_encoder = models.resnet18(weights=None)
        
        # 修改第一层卷积适应 3/4/5 通道
        self.vision_encoder.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.vision_encoder.fc = nn.Identity() 
        vision_feature_dim = 512
        
        # === 2. 策略头 (输出当前单步动作) ===
        # 输入: 512
        # 输出: 2 (v, w)
        self.policy_head = nn.Sequential(
            nn.Linear(vision_feature_dim, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, action_dim) # 只输出 [v, w]
        )

    def forward(self, image):
        """
        Output: [Batch, 2]
        """
        # 1. 提取特征
        vision_feat = self.vision_encoder(image) # [B, 512]
        
        # 2. 回归预测
        action_pred = self.policy_head(vision_feat) # [B, 2]
        
        return action_pred