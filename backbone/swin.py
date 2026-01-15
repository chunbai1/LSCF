import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from .utils import PatchEmbed, BasicLayer, PatchMerging, load_checkpoint



class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        with_cp (bool): Whether to use checkpointing to save memory. Default: False.
        vlf_ris: Vision language fusion for referring image segmentation. Support 'LAVT', 'LGCE', 'RMSIN'. Default: None.
        num_heads_fusion, fusion_drop: Parameters for vlf_ris.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 with_cp=False,
                 l_dim=768,
                 vlf_ris=None,
                 num_heads_fusion=[1, 1, 1, 1],
                 fusion_drop=0.,
                 vlf_vg=None,
                 fianet_num_tmem=1):
        super().__init__()

        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.l_dim = l_dim
        self.vlf_ris = vlf_ris
        self.vlf_vg = vlf_vg

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                with_cp=with_cp,
                l_dim=l_dim,
                vlf_ris=None if (self.vlf_ris == 'RefSegformer' and i_layer == 0) else self.vlf_ris,
                num_heads_fusion=num_heads[i_layer] if self.vlf_ris == 'RefSegformer' else num_heads_fusion[i_layer],
                fusion_drop=fusion_drop,
                vlf_vg=self.vlf_vg,
                size=img_size // (2 ** (i_layer + 2)),
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # if self.vlf_ris == 'RMSIN':
        #     self.cim = CIM(dim=sum(num_features), channels=num_features,
        #                    height=img_size//32, width=img_size//32)
        # elif self.vlf_ris == 'SLViT':
        #     self.urce = URCE(channels=num_features, dim=256)
        #     self.squeelayers = nn.ModuleList()
        #     self.normlayers = nn.ModuleList()
        #     self.act = nn.ReLU(inplace=True)
        #     for i in range(self.num_layers):
        #         self.squeelayers.append(
        #             nn.Conv2d(num_features[i]*2, num_features[i], 1, 1)
        #         )
        #         self.normlayers.append(
        #             nn.LayerNorm(num_features[i])
        #         )
        # elif self.vlf_ris == 'FIANet':
        #     self.tmem = TMEM(dim=sum(num_features), num_blocks=fianet_num_tmem, channels=num_features)
        # if self.vlf_ris == 'MSCF':
        #     self.vli = VLI(num_features, num_features)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, l=None, l_mask=None, t=None, t_mask=None, p=None, p_mask=None):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        sim_temps = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww, sim_temp = layer(x, Wh, Ww, l, l_mask, t, t_mask, p, p_mask)
            if self.vlf_ris == 'DMMI' or self.vlf_ris == 'DIVL':
                l = sim_temp

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                sim_temps.append(sim_temp)

        # if self.vlf_ris == 'RMSIN':
        #     outs = self.cim(outs)
        # elif self.vlf_ris == 'SLViT':
        #     features = outs
        #     feature_trans = self.urce(features, sim_temps)
        #     outs = []
        #     for i in range(self.num_layers):
        #         skip = self.squeelayers[i](torch.cat((feature_trans[i], features[i]), dim=1))
        #         skip = self.act(self.normlayers[i](skip.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous())
        #         outs.append(skip)
        # elif self.vlf_ris == 'FIANet':
        #     outs = self.tmem(outs, l, l_mask)
        # if self.vlf_ris == 'MSCF':
        #     outs = self.vli(outs, l, l_mask)

        # if self.vlf_ris == 'DMMI' or self.vlf_ris == 'DIVL':
        #     return tuple(outs), l
        # else:
        return tuple(outs)
