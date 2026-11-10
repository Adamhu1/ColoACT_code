import torch
import torch.nn as nn
import torchvision.models as models

class ColonBCSingleStep(nn.Module):
    def __init__(self, action_dim=2, input_channels=5):
        super().__init__()
        
        # === 1. 瑙嗚缂栫爜鍣?(淇濇寔涓?Diffusion/Chunking BC 涓€鑷? ===
        # 蹇呴』鐢ㄤ竴鏍风殑楠ㄥ共缃戠粶锛岀‘淇濆叕骞?
        self.vision_encoder = models.resnet18(weights=None)
        
        # 淇敼绗竴灞傚嵎绉€傚簲 3/4/5 閫氶亾
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
        
        # === 2. 绛栫暐澶?(杈撳嚭褰撳墠鍗曟鍔ㄤ綔) ===
        # 杈撳叆: 512
        # 杈撳嚭: 2 (v, w)
        self.policy_head = nn.Sequential(
            nn.Linear(vision_feature_dim, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, action_dim) # 鍙緭鍑?[v, w]
        )

    def forward(self, image):
        """
        Output: [Batch, 2]
        """
        # 1. 鎻愬彇鐗瑰緛
        vision_feat = self.vision_encoder(image) # [B, 512]
        
        # 2. 鍥炲綊棰勬祴
        action_pred = self.policy_head(vision_feat) # [B, 2]
        
        return action_pred


