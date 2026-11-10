import torch
import torch.nn as nn
import torchvision.models as models

class ColonBCPolicy(nn.Module):
    def __init__(self, action_dim=2, pred_horizon=16, input_channels=5):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        
        # === 1. 瑙嗚缂栫爜鍣?(蹇呴』涓?Diffusion 淇濇寔瀹屽叏涓€鑷? ===
        # 杩欐牱鎵嶈兘璇佹槑鏄?Diffusion 绠楁硶寮猴紝鑰屼笉鏄?ResNet 寮?
        self.vision_encoder = models.resnet18(weights=None)
        
        # 淇敼绗竴灞傚嵎绉互閫傚簲 3/4/5 閫氶亾
        self.vision_encoder.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        # ResNet18 fc 杈撳嚭鏄?1000锛屾垜浠敼鎴?Identity锛屽彇 avgpool 涔嬪墠鐨勭壒寰?
        # 鎴栬€呯洿鎺ョ敤 fc 杈撳嚭 512
        self.vision_encoder.fc = nn.Identity() 
        
        # ResNet18 鏈€鍚庝竴灞?avgpool 鍑烘潵鐨勭壒寰佹暟鏄?512
        vision_feature_dim = 512
        
        # === 2. 绛栫暐澶?(绠€鍗曠殑 MLP) ===
        # 杈撳叆: 瑙嗚鐗瑰緛 (512)
        # 杈撳嚭: 鍔ㄤ綔搴忓垪 (16 * 2 = 32)
        self.policy_head = nn.Sequential(
            nn.Linear(vision_feature_dim, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, pred_horizon * action_dim) # 鐩存帴杈撳嚭鎵€鏈夋鏁扮殑鍔ㄤ綔
        )

    def forward(self, image):
        """
        璁粌/鎺ㄧ悊 閫氱敤
        Output: [Batch, 16, 2]
        """
        # 1. 鎻愬彇鐗瑰緛
        vision_feat = self.vision_encoder(image) # [B, 512]
        
        # 2. 鍥炲綊棰勬祴
        action_flat = self.policy_head(vision_feat) # [B, 32]
        
        # 3. Reshape 鎴愬簭鍒?[B, 16, 2]
        action_pred = action_flat.view(-1, self.pred_horizon, self.action_dim)
        
        return action_pred


