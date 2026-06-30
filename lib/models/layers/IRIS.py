
import torch
import torch.nn as nn

class IRIS(nn.Module):
    def __init__(self, embedding_dim):
        super(IRIS, self).__init__()

        # 用 RGB+TIR 联合生成 λ（更符合“门控谁说了算”的语义）
        self.lambda_conv = nn.Sequential(
            nn.Conv1d(2 * embedding_dim, embedding_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_dim // 2, 1, kernel_size=3, padding=1),
        )
        # 学习型温度，控制门的“硬度”（>1 变硬，<1 变软）
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # 通道注意力 CA
        self.conv_ca  = nn.Conv1d(embedding_dim, embedding_dim // 2, kernel_size=1)
        self.conv_ca2 = nn.Conv1d(embedding_dim // 2, embedding_dim, kernel_size=1)
        self.sigmoid  = nn.Sigmoid()

        # 空间注意力 SA
        self.conv_sa = nn.Conv1d(2, 1, kernel_size=3, padding=1)

    def get_lambda(self, x_rgb, x_tir):
        """用 RGB+TIR 生成 Patch 级 λ；返回 [B, N, 1] 和未激活的 logit"""
        # [B, N, C] -> [B, C, N]
        x_rgb = x_rgb.permute(0, 2, 1)
        x_tir = x_tir.permute(0, 2, 1)
        g_in  = torch.cat([x_rgb, x_tir], dim=1)      # [B, 2C, N]
        lam_logit = self.lambda_conv(g_in)            # [B, 1, N]
        lam = torch.sigmoid(lam_logit * self.temperature)   # 温度锐化后的 λ
        lam = lam.permute(0, 2, 1)                    # [B, N, 1]
        lam_logit = lam_logit.permute(0, 2, 1)        # [B, N, 1]
        return lam, lam_logit

    def channel_attention(self, x):
        """通道注意力 (CA)，x: [B, C, N]"""
        avg_pool = torch.mean(x, dim=2, keepdim=True)  # [B, C, 1]
        ca_weight = self.conv_ca2(self.conv_ca(avg_pool))
        ca_weight = self.sigmoid(ca_weight)            # [B, C, 1]
        return x * ca_weight                           # [B, C, N]

    def spatial_attention(self, x):
        """空间注意力 (SA)，x: [B, C, N]"""
        avg_out = torch.mean(x, dim=1, keepdim=True)     # [B, 1, N]
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # [B, 1, N]
        sa_weight = self.sigmoid(self.conv_sa(torch.cat([avg_out, max_out], dim=1)))  # [B, 1, N]
        return x * sa_weight                             # [B, C, N]

    def forward(self, rgb_feat, tir_feat, return_lambda=False):
        """
        rgb_feat, tir_feat: [B, N, C]
        return:
            fused: [B, N, C]
            (optional) lambda_tokens: [B, N, 1]
        """
        lambda_factor, lambda_logit = self.get_lambda(rgb_feat, tir_feat)  # [B, N, 1]

        # 模态加权融合
        rgb_weighted = rgb_feat * lambda_factor
        tir_weighted = tir_feat * (1 - lambda_factor)
        fused = rgb_weighted + tir_weighted  # [B, N, C]

        # CA + SA
        fused = self.channel_attention(fused.permute(0, 2, 1))  # [B, C, N]
        fused = self.spatial_attention(fused)                   # [B, C, N]
        fused = fused.permute(0, 2, 1)                          # [B, N, C]

        if return_lambda:
            return fused, lambda_factor
        return fused