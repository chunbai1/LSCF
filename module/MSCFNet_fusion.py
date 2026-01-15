import torch
from torch import nn
from .utils import MSDC, CrossModalAttention
from .utils import EfficientAgentAttention, MLP


class MSCFNet_fusion(nn.Module):
    def __init__(self, dim, l_dim, num_heads=8, dropout=0.1, size=16, mlp_ratio=4.):
        super(MSCFNet_fusion, self).__init__()
        self.kernel_size = [1, 3, 5]
        self.msdc = MSDC(dim, self.kernel_size, stride=1)

        self.vis_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
        )

        self.lang_proj = nn.Sequential(
            nn.Linear(l_dim, l_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm_ca_vis = nn.LayerNorm(dim)
        self.norm_ca_lang = nn.LayerNorm(l_dim)
        self.attn = CrossModalAttention(dim, l_dim, dim, dim, dim, num_heads)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim*4, dim, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False),
        )

        self.norm_sa_mm = nn.LayerNorm(dim)
        self.norm_sa_mm_lang = nn.LayerNorm(dim)
        self.fusion_attn = EfficientAgentAttention(dim, num_heads, dropout, size)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = MLP(dim, int(dim * mlp_ratio))

        self.res_gate = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.Tanh()
        )

    def forward(self, x, l, l_mask):
        B, HW, C = x.shape
        H = W = int(HW ** 0.5)

        x_reshape = x.permute(0, 2, 1).reshape(B, C, H, W)
        outs = self.msdc(x_reshape)

        x_proj = self.vis_proj(x_reshape).flatten(2).permute(0, 2, 1)
        l_proj = self.lang_proj(l.permute(0, 2, 1))
        ca_out = x + self.attn(self.norm_ca_vis(x_proj), self.norm_ca_lang(l_proj).permute(0, 2, 1), l_mask)

        outs.append((ca_out).permute(0, 2, 1).reshape(B, C, H, W))
        x_res = self.fusion(torch.concat((outs), dim=1)).flatten(2).permute(0, 2, 1)
        
        e_x_res = x_res + self.fusion_attn(self.norm_sa_mm(x_res), self.norm_sa_mm_lang(ca_out))
        out = e_x_res + self.ffn(self.norm_ffn(e_x_res))
        x = x + (self.res_gate(out) * out)

        return x, out
