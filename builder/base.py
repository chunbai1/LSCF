import importlib
from torch import nn
from builder.model_info import *


class BaseEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super(BaseEncoderDecoder, self).__init__()
        self.cfg = cfg
        self.enc_cfg, self.dec_cfg = self._base_init(cfg)

        self.backbone = self._build_backbone()
        self.decoder = self._build_decoder()

    def _base_init(self, cfg):
        model_cfg = cfg['model']
        enc_cfg = model_cfg.get('backbone')
        dec_cfg = model_cfg.get('decoder')

        return enc_cfg, dec_cfg

    def _build_backbone(self):
        self.enc_cfg['kwargs']['img_size'] = self.cfg.get(
            'crop_size', 480)

        if self.cfg['model'].get('text_encoder', False):
            self.enc_cfg['kwargs']['l_dim'] = language_info[self.cfg['model']['text_encoder']['type']]['hidden_dim']

        encoder = self._build_enc_module(self.enc_cfg['type'], self.enc_cfg.get('kwargs'))
        encoder.init_weights(self.enc_cfg.get('pretrained'))

        return encoder

    def _build_decoder(self):
        dec_type, dec_kwargs = self.dec_cfg.get('type'), self.dec_cfg.get('kwargs', {})
        assert dec_type != None, 'What are you doing ? You must set the decoder type !!!'

        dec_kwargs['embedding_dim'] = decoder_info[dec_type]['dec_out_dim']
        dec_kwargs['num_classes'] = 2

        # dec_kwargs['in_channels'] = encoder_info[self.enc_cfg['type']]['enc_out_dims']
        dec_kwargs['in_channels'] = encoder_info[self.enc_cfg['type']]['enc_out_dims']

        decoder = self._build_dec_module(dec_type, dec_kwargs)
        return decoder

    def _build_enc_module(self, mtype, kwargs):
        enc = getattr(importlib.import_module('backbone'), mtype)
        return enc(**kwargs)

    def _build_dec_module(self, mtype, kwargs):
        dec = getattr(importlib.import_module('decoder'), mtype)
        return dec(**kwargs)

    def base_forward(self, x):
        pass

    def tta_forward(self, x):
        pass

    def forward(self, x, tta=False):
        if not tta:
            return self.base_forward(x)
        else:
            return self.tta_forward(x)