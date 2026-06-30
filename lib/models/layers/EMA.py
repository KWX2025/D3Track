import torch
from torch import nn
#GitHub地址：https://github.com/YOLOonMe/EMA-attention-module
#论文地址：https://arxiv.org/abs/2305.13563v2
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, n, c = x.size()  # 输入形状 [B, N, C]

        # 计算最近的完全平方数
        h = w = math.ceil(math.sqrt(n))
        new_n = h * w

        # 填充到最近的完全平方数
        if new_n > n:
            pad_size = new_n - n
            x = torch.cat([x, torch.zeros(b, pad_size, c, device=x.device)], dim=1)  # 填充

        # 将 [B, N, C] 转换为 [B, C, H, W]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)

        # 将 [B, C, H, W] 转回 [B, N, C] 并去除填充
        out = out.reshape(b, c, -1).permute(0, 2, 1)  # [B, N, C]
        if new_n > n:
            out = out[:, :n, :]  # 去掉填充部分
        return out

# 测试代码
if __name__ == '__main__':
    block = EMA(64).cuda()
    input_tensor = torch.rand(1, 320, 64).cuda()  # 示例输入 [B, N, C], N=320, C=64
    output_tensor = block(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
