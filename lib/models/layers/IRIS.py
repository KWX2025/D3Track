
import torch
import torch.nn as nn

class IRIS(nn.Module):
    def __init__(self, embedding_dim):
        super(IRIS, self).__init__()

        
        self.lambda_conv = nn.Sequential(
            nn.Conv1d(2 * embedding_dim, embedding_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_dim // 2, 1, kernel_size=3, padding=1),
        )
        
        self.temperature = nn.Parameter(torch.tensor(1.0))

        
        self.conv_ca  = nn.Conv1d(embedding_dim, embedding_dim // 2, kernel_size=1)
        self.conv_ca2 = nn.Conv1d(embedding_dim // 2, embedding_dim, kernel_size=1)
        self.sigmoid  = nn.Sigmoid()

        
        self.conv_sa = nn.Conv1d(2, 1, kernel_size=3, padding=1)

    def get_lambda(self, x_rgb, x_tir):
        
        
        x_rgb = x_rgb.permute(0, 2, 1)
        x_tir = x_tir.permute(0, 2, 1)
        g_in  = torch.cat([x_rgb, x_tir], dim=1)      
        lam_logit = self.lambda_conv(g_in)            
        lam = torch.sigmoid(lam_logit * self.temperature)   
        lam = lam.permute(0, 2, 1)                    
        lam_logit = lam_logit.permute(0, 2, 1)        
        return lam, lam_logit

    def channel_attention(self, x):
        
        avg_pool = torch.mean(x, dim=2, keepdim=True)  
        ca_weight = self.conv_ca2(self.conv_ca(avg_pool))
        ca_weight = self.sigmoid(ca_weight)            
        return x * ca_weight                           

    def spatial_attention(self, x):
        
        avg_out = torch.mean(x, dim=1, keepdim=True)     
        max_out, _ = torch.max(x, dim=1, keepdim=True)   
        sa_weight = self.sigmoid(self.conv_sa(torch.cat([avg_out, max_out], dim=1)))  
        return x * sa_weight                             

    def forward(self, rgb_feat, tir_feat, return_lambda=False):
        
        lambda_factor, lambda_logit = self.get_lambda(rgb_feat, tir_feat)  

        
        rgb_weighted = rgb_feat * lambda_factor
        tir_weighted = tir_feat * (1 - lambda_factor)
        fused = rgb_weighted + tir_weighted  

        
        fused = self.channel_attention(fused.permute(0, 2, 1))  
        fused = self.spatial_attention(fused)                   
        fused = fused.permute(0, 2, 1)                          

        if return_lambda:
            return fused, lambda_factor
        return fused