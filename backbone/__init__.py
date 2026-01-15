from .swin import SwinTransformer


__all__ = ['swin_tiny', 'swin_small', 'swin_base', 'swin_large', 'swin_base_gfm_w6']



class swin_tiny(SwinTransformer):
    def __init__(self, **kwargs):
        super(swin_tiny, self).__init__(
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], **kwargs)


class swin_small(SwinTransformer):
    def __init__(self, **kwargs):
        super(swin_small, self).__init__(
            embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], **kwargs)


class swin_base(SwinTransformer):
    def __init__(self, **kwargs):
        super(swin_base, self).__init__(
            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], **kwargs)


class swin_large(SwinTransformer):
    def __init__(self, **kwargs):
        super(swin_large, self).__init__(
            embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], **kwargs)


class swin_base_gfm_w6(SwinTransformer):
    def __init__(self, **kwargs):
        super(swin_base_gfm_w6, self).__init__(
            window_size=6, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], **kwargs)