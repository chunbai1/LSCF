import torch
from torch import nn
from torch.nn import functional as F
from .FreqFusion import FreqFusion
from .utils import CLIM, CBR

class FreqLAVTHead(nn.Module):
    '''
    from 'LAVT: Language-Aware Vision Transformer for Referring Image Segmentation'
    '''
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 embedding_dim=512,
                 num_classes=2,
                 **kwargs):
        super(FreqLAVTHead, self).__init__()

        # embedding_dim = in_channels[-2]
        c1_size, c2_size, c3_size, c4_size = in_channels

        self.x4_proj = CBR(c4_size, embedding_dim, 1, 1)
        self.x3_proj = CBR(c3_size, embedding_dim, 1, 1)
        self.x2_proj = CBR(c2_size, embedding_dim, 1, 1)
        self.x1_proj = CBR(c1_size, embedding_dim, 1, 1)

        self.clim4 = CLIM(embedding_dim, 768, embedding_dim)
        self.post_proj4 = CBR(embedding_dim, embedding_dim, 1, 1)
        
        self.freq_mix3 = FreqFusion(embedding_dim, embedding_dim)
        self.freq_proj3 = CBR(embedding_dim*2, embedding_dim, 3, 1, 1)
        self.clim3 = CLIM(embedding_dim, 768, embedding_dim)
        self.post_proj3 = CBR(embedding_dim, embedding_dim, 1, 1)

        self.freq_mix2 = FreqFusion(embedding_dim, embedding_dim)
        self.freq_proj2 = CBR(embedding_dim*2, embedding_dim, 3, 1, 1)
        self.clim2 = CLIM(embedding_dim, 768, embedding_dim)
        self.post_proj2 = CBR(embedding_dim, embedding_dim, 1, 1)

        self.freq_mix1 = FreqFusion(embedding_dim, embedding_dim)
        self.freq_proj1 = CBR(embedding_dim*2, embedding_dim, 3, 1, 1)
        self.clim1 = CLIM(embedding_dim, 768, embedding_dim)
        self.post_proj1 = CBR(embedding_dim, embedding_dim, 1, 1)

        self.conv1_1 = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, inputs, l, l_mask):
        l = l.permute(0, 2, 1)
        x_c1, x_c2, x_c3, x_c4 = inputs

        x4 = self.x4_proj(x_c4)
        x3 = self.x3_proj(x_c3)
        x2 = self.x2_proj(x_c2)
        x1 = self.x1_proj(x_c1)

        x4 = self.clim4(x4, l, l_mask)
        x4 = self.post_proj4(x4)

        _, x3, x4_up = self.freq_mix3(x3, x4)
        # x4_up = F.interpolate(input=x4, size=(x3.size(-2), x3.size(-1)), mode='bilinear', align_corners=True)
        x43 = self.freq_proj3(torch.cat([x3, x4_up], dim=1))
        x43 = self.clim3(x43, l, l_mask)
        x43 = self.post_proj3(x43)

        _, x2, x43_up = self.freq_mix3(x2, x43)
        # x43_up = F.interpolate(input=x43, size=(x2.size(-2), x2.size(-1)), mode='bilinear', align_corners=True)
        x432 = self.freq_proj2(torch.cat([x2, x43_up], dim=1))
        x432 = self.clim2(x432, l, l_mask)
        x432 = self.post_proj2(x432)

        _, x1, x432_up = self.freq_mix3(x1, x432)
        # x432_up = F.interpolate(input=x432, size=(x1.size(-2), x1.size(-1)), mode='bilinear', align_corners=True)
        x4321 = self.freq_proj1(torch.cat([x1, x432_up], dim=1))
        x4321 = self.clim1(x4321, l, l_mask)
        x4321 = self.post_proj1(x4321)

        return self.conv1_1(x4321)
