import torch
import torch.nn as nn
import torchvision.models as models


#BC鐢ㄧ殑model

class ColonNavNet(nn.Module):
    def __init__(self, action_dim=2):
        super(ColonNavNet, self).__init__()
        
        # 浣跨敤 ResNet18 浣滀负鐗瑰緛鎻愬彇鍣?(杞婚噺涓旀湁鏁?
        # weights=None 琛ㄧず涓嶄娇鐢ㄩ璁粌鏉冮噸锛屽洜涓虹粨鑲犳暟鎹笌ImageNet宸紓宸ㄥぇ锛?
        # 涓旀垜浠殑杈撳叆閫氶亾鏁颁笉鍚岋紝寤鸿浠庡ご璁粌銆?
        self.backbone = models.resnet18(weights=None)
        
        # 淇敼绗竴灞傚嵎绉眰锛屼互閫傚簲鎴戜滑鐨勮緭鍏ラ€氶亾鏁?
        # 鍘熷 ResNet 杈撳叆鏄?3 (RGB)锛屾垜浠渶瑕佹敼涓?2 (娣卞害鍥?+ 楂樼▼鍥?
        # 鐢氳嚦鍙互鏀逛负 3 (娣卞害 + 楂樼▼ + 鍘熷浘鐏板害)锛岃繖閲屽厛鎸変綘鐨勮姹傜敤 2
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # 淇敼鍏ㄨ繛鎺ュ眰 (FC)锛岃緭鍑哄姩浣滅淮搴?(v, w)
        # ResNet18 鐨?fc 杈撳叆鐗瑰緛鏁版槸 512
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # 闃叉杩囨嫙鍚?
            nn.Linear(256, action_dim) # 杈撳嚭 [v, w]
        )

    def forward(self, x):
        # x shape: [batch, 2, 240, 320]
        return self.backbone(x)

if __name__ == '__main__':
    # 娴嬭瘯妯″瀷缁撴瀯
    net = ColonNavNet()
    dummy_input = torch.randn(1, 2, 240, 320) # [Batch, Channels, H, W]
    output = net(dummy_input)
    print(f"Output shape: {output.shape}") # 搴旇鏄?[1, 2]


