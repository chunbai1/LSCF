from torch import nn
import torch
import torch.nn.functional as F

"""source"""
class CLIM(nn.Module):
    def __init__(self, v_inp_dim, t_inp_dim, embed_dim):
        super(CLIM, self).__init__()


        self.embed_dim = embed_dim
        self.v_trans = nn.Sequential(
            nn.Conv2d(v_inp_dim, self.embed_dim, 1),
            nn.Tanh(),
        )
        self.t_trans = nn.Sequential(
            nn.Linear(t_inp_dim, self.embed_dim),
            nn.Tanh(),
        )
        self.f_out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, v_inp_dim, 1),
            nn.BatchNorm2d(v_inp_dim),
            nn.ReLU()
        )


    def forward(self, v_feat, t_feat, t_mask):
        vis_feats = self.v_trans(v_feat)
        lang_feats = self.t_trans(t_feat)
        sent_feat = torch.div(torch.sum(lang_feats, 1), torch.sum(t_mask.float(),1)).unsqueeze(2).unsqueeze(3)
        vis_feats = self.f_out(vis_feats * sent_feat.expand_as(vis_feats))
        vis_feats = F.normalize(v_feat + vis_feats, p=2, dim=1)
        return vis_feats


class CBR(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 with_residual=False):
        super(CBR, self).__init__()
        self.with_residual = with_residual

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        if not self.with_residual:
            out = self.act(self.norm(self.conv(x)))
        else:
            out = x + self.act(self.norm(self.conv(x)))

        return out


# class CLIM(nn.Module):
#     def __init__(self, v_inp_dim, t_inp_dim, embed_dim):
#         super(CLIM, self).__init__()


#         self.embed_dim = embed_dim
#         self.v_trans = nn.Sequential(
#             nn.Conv2d(v_inp_dim, self.embed_dim, 1),
#             nn.GroupNorm(32, self.embed_dim)
#             # nn.BatchNorm2d(self.embed_dim),
#             # nn.ReLU(inplace=True)
#         )
#         self.t_trans = nn.Sequential(
#             nn.Linear(t_inp_dim, self.embed_dim),
#             nn.Tanh(),
#         )
#         self.f_out = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(self.embed_dim, v_inp_dim, 1),
#             nn.BatchNorm2d(v_inp_dim),
#             nn.ReLU()
#         )


#     def forward(self, v_feat, t_feat, t_mask):
#         vis_feats = self.v_trans(v_feat)
#         lang_feats = self.t_trans(t_feat)
#         sent_feat = torch.div(torch.sum(lang_feats, 1), torch.sum(t_mask.float(),1)).unsqueeze(2).unsqueeze(3)
#         # sent_feat = torch.mean(lang_feats, dim=1).unsqueeze(2).unsqueeze(3)
#         vis_feats = self.f_out(vis_feats * sent_feat.expand_as(vis_feats))
#         vis_feats = v_feat * vis_feats
#         return vis_feats