import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class SinusoidalPosEmb(nn.Module):
    """
    标准的正弦位置编码：让 Transformer 知道动作的前后顺序
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ACTPolicy(nn.Module):
    def __init__(self, pred_horizon=16, obs_horizon=1, action_dim=2, input_channels=5, latent_dim=32, dim_model=256):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.dim_model = dim_model

        # ==========================
        # 1. 视觉主干 (Backbone)
        # ==========================
        # 对于内窥镜导航，ResNet18 足够且比 DETR 更适合全局特征提取
        self.backbone = models.resnet18(weights=None) # 如果有 ImageNet 预训练权重更好，没有也行
        # 修改第一层卷积以适应 input_channels
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity() # 去掉最后分类层，输出 [B, 512]
        
        # 投影到 Transformer 的维度
        self.visual_proj = nn.Linear(512, dim_model)

        # ==========================
        # 2. CVAE Encoder (Posterior)
        # ==========================
        # 这里的任务是：看图片 + 看未来动作 -> 压缩成 z
        # Encoder 输入: [CLS, Visual, Action_0, ..., Action_K]
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))
        
        # 动作编码器
        self.action_encoder = nn.Linear(action_dim, dim_model)
        
        # 位置编码 (关键升级！)
        self.pos_encoder = SinusoidalPosEmb(dim_model)

        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=4, dim_feedforward=1024, batch_first=True, dropout=0.1)
        self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 预测均值和对数方差
        self.latent_proj = nn.Linear(dim_model, latent_dim * 2) 

        # ==========================
        # 3. Policy Decoder (Prior / Generator)
        # ==========================
        # 这里的任务是：看图片 + z -> 生成未来动作
        # 输入: [z, Visual, Pos_Query_0, ..., Pos_Query_K]
        
        self.latent_out_proj = nn.Linear(latent_dim, dim_model)
        
        # 固定的位置查询 (Fixed Position Queries)
        # 告诉 Decoder 我们要预测的是第 0 到 第 15 步
        self.pos_queries = nn.Embedding(pred_horizon, dim_model)
        
        # Decoder Transformer
        # 注意：原文虽然叫 Decoder，但在 PyTorch 实现中通常用 Encoder 结构堆叠即可（因为是自回归或并行生成）
        decoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=4, dim_feedforward=1024, batch_first=True, dropout=0.1)
        self.policy_decoder = nn.TransformerEncoder(decoder_layer, num_layers=4) # 更深一点

        # 最后的动作头
        self.action_head = nn.Linear(dim_model, action_dim)

    def forward(self, obs, actions=None):
        """
        Input:
            obs: [B, C, H, W]
            actions: [B, pred_horizon, action_dim] (Training only)
        Output:
            pred_actions: [B, pred_horizon, action_dim]
            mu, logvar: Latent params
        """
        B = obs.shape[0]
        device = obs.device

        # 1. 视觉特征提取
        visual_feat = self.backbone(obs) # [B, 512]
        visual_emb = self.visual_proj(visual_feat).unsqueeze(1) # [B, 1, dim_model]

        # ==========================
        # CVAE 训练逻辑 (Training)
        # ==========================
        mu, logvar = None, None
        if actions is not None:
            # 构建 Encoder 输入序列: [CLS, Visual, Action_Sequence...]
            
            # (1) 动作嵌入
            actions_emb = self.action_encoder(actions) # [B, T, dim_model]
            
            # (2) 添加位置编码 (这是之前缺少的！)
            # 生成 0 ~ T-1 的位置索引
            pos_ids = torch.arange(self.pred_horizon, device=device).float()
            pos_emb = self.pos_encoder(pos_ids) # [T, dim_model]
            actions_emb = actions_emb + pos_emb.unsqueeze(0) # 广播相加

            # (3) 拼接
            cls_tokens = self.cls_token.expand(B, -1, -1)
            encoder_in = torch.cat([cls_tokens, visual_emb, actions_emb], dim=1) # [B, 1+1+T, dim]
            
            # (4) 过 Transformer
            enc_out = self.cvae_encoder(encoder_in)
            
            # (5) 只取 CLS token 预测 z
            cls_out = enc_out[:, 0, :] 
            latent_params = self.latent_proj(cls_out)
            mu = latent_params[:, :self.latent_dim]
            logvar = latent_params[:, self.latent_dim:]
            
            # 重参数化采样
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # 推理时，z 设为 0 (确定性模式) 或从标准正态分布采样
            z = torch.zeros((B, self.latent_dim), device=device)

        # ==========================
        # 策略生成逻辑 (Generation)
        # ==========================
        # 构建 Decoder 输入: [z, Visual, Pos_Queries...]
        
        # (1) 处理 z
        z_emb = self.latent_out_proj(z).unsqueeze(1) # [B, 1, dim]
        
        # (2) 准备查询位置 (Queries)
        # 我们需要预测 16 个动作，所以准备 16 个 Query Slot
        query_ids = torch.arange(self.pred_horizon, device=device).long()
        query_emb = self.pos_queries(query_ids).unsqueeze(0).expand(B, -1, -1) # [B, T, dim]
        
        # (3) 拼接输入: z + visual + queries
        # 这样 Transformer 就能同时看到：整体风格(z) + 当前画面(visual) + 我要填第几个空(query)
        decoder_in = torch.cat([z_emb, visual_emb, query_emb], dim=1)
        
        # (4) 过 Transformer
        # 这里需要注意 Mask，防止看到未来的信息？
        # 对于 ACT，我们是一次性生成整个序列，所以不需要 Causal Mask，用双向 Attention 即可
        dec_out = self.policy_decoder(decoder_in)
        
        # (5) 取出对应的部分 (去掉 z 和 visual，只留后面 T 个 Query 的输出)
        action_out = dec_out[:, 2:, :] # 跳过前两个 token (z, visual)
        
        pred_actions = self.action_head(action_out) # [B, T, action_dim]

        return pred_actions, mu, logvar