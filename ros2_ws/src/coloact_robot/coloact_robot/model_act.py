import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ACTPolicy(nn.Module):
    def __init__(self, pred_horizon=16, obs_horizon=1, action_dim=2, input_channels=5, latent_dim=32):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # 1. 瑙嗚缂栫爜鍣?(ResNet18)
        # 淇敼绗竴灞備互閫傚簲 input_channels (RGB=3, RGBD=4, RGBD+Elev=5)
        self.vision_encoder = models.resnet18(pretrained=False)
        self.vision_encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.vision_encoder.fc = nn.Identity() # 鍘绘帀鏈€鍚庡垎绫诲眰
        # ResNet18 杈撳嚭 512 缁寸壒寰?
        
        # 2. CVAE Encoder (璁粌鏃剁敤: Obs + Action -> z)
        # 杈撳叆: [Vision_Feat (512) + Action_Seq (16*2)]
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim)) # 鐢ㄤ簬鎻愬彇 z
        self.encoder_input_proj = nn.Linear(512 + pred_horizon * action_dim, 256)
        
        # 绠€鍗曠殑 Transformer Encoder 妯℃嫙 CVAE 缂栫爜杩囩▼
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 鎶曞奖鍒板潎鍊煎拰鏂瑰樊
        self.latent_proj = nn.Linear(256, latent_dim * 2) # 鍓嶄竴鍗?mu, 鍚庝竴鍗?logvar

        # 3. CVAE Decoder (鎺ㄧ悊鏃剁敤: Obs + z -> Action)
        self.decoder_input_proj = nn.Linear(512 + latent_dim, 256)
        # 绠€鍗曠殑 Transformer Decoder 鐢熸垚搴忓垪
        decoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=3)
        
        self.action_head = nn.Linear(256, action_dim)

    def forward(self, obs, actions=None):
        """
        璁粌妯″紡: 杈撳叆 obs 鍜?actions, 杩斿洖 reconstruction, mu, logvar
        鎺ㄧ悊妯″紡: 杈撳叆 obs, 杩斿洖 actions (z 璁句负 0)
        """
        B = obs.shape[0]
        
        # 1. 鎻愬彇瑙嗚鐗瑰緛 [B, 512]
        vision_feat = self.vision_encoder(obs)
        
        if actions is not None:
            # === Training (CVAE Forward) ===
            # Flatten actions: [B, 16, 2] -> [B, 32]
            actions_flat = actions.view(B, -1)
            
            # 鎷兼帴: [Vision, Actions]
            encoder_in = torch.cat([vision_feat, actions_flat], dim=1) # [B, 512+32]
            encoder_emb = self.encoder_input_proj(encoder_in).unsqueeze(1) # [B, 1, 256]
            
            # 缂栫爜
            enc_out = self.encoder_transformer(encoder_emb)
            # 棰勬祴 mu, logvar
            latent_params = self.latent_proj(enc_out.squeeze(1))
            mu = latent_params[:, :self.latent_dim]
            logvar = latent_params[:, self.latent_dim:]
            
            # 閲嶅弬鏁板寲閲囨牱 (Reparameterization Trick)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # === Inference (Test time) ===
            # 鎺ㄧ悊鏃讹紝閫氬父鐩存帴璁?z=0 (鍧囧€? 鏉ヨ幏寰楃‘瀹氭€ц緭鍑猴紝鎴栬€呴殢鏈洪噰鏍?
            mu = torch.zeros((B, self.latent_dim)).to(obs.device)
            logvar = torch.zeros((B, self.latent_dim)).to(obs.device)
            z = mu 

        # === Decoder ===
        # 鎷兼帴: [Vision, z]
        decoder_in = torch.cat([vision_feat, z], dim=1) # [B, 512+32]
        decoder_emb = self.decoder_input_proj(decoder_in).unsqueeze(1) # [B, 1, 256]
        
        # 鎴戜滑闇€瑕佺敓鎴?pred_horizon 涓椂闂存
        # 杩欓噷鐢ㄤ竴涓皬 Trick: 閲嶅杈撳叆骞堕€氳繃 Transformer 缁嗗寲
        decoder_emb = decoder_emb.repeat(1, self.pred_horizon, 1) # [B, 16, 256]
        
        # 娣诲姞浣嶇疆缂栫爜 (绠€鍗曠殑 Sinusoidal PE 鍙互鐪佺暐锛屽洜涓哄簭鍒楀緢鐭紝MLP灏辫兘瀛︿細浣嶇疆)
        # 杩欓噷鐩存帴杩?Transformer
        dec_out = self.decoder_transformer(decoder_emb) # [B, 16, 256]
        
        pred_actions = self.action_head(dec_out) # [B, 16, 2]
        
        return pred_actions, mu, logvar


