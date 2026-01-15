import torch.nn.functional as F
from .base import BaseEncoderDecoder
from language import BertModel
from builder.model_info import *


class ris_model_builder(BaseEncoderDecoder):
    def __init__(self, cfg):
        super(ris_model_builder, self).__init__(cfg)

        self.text_enc_type = cfg['model']['text_encoder'].get('type', 'bert-base-uncased')
        self.text_encoder = self.init_text_enc(self.text_enc_type)

    def base_forward(self, x, l, l_mask, t_mask=None, p_mask=None):
        h, w = x.shape[-2:]

        l_feats, l_mask, t_feats, t_mask, p_feats, p_mask = self.language_forward(self.text_enc_type, l, l_mask, t_mask=t_mask, p_mask=p_mask)
        feats = self.backbone(x, l_feats, l_mask, t_feats, t_mask, p_feats, p_mask)
        out = self.various_decoder(self.dec_cfg['type'], feats, l_feats, l_mask, h, w)

        return out


    def forward(self, x, l, l_mask, t_mask=None, p_mask=None, tta=False):
        if not tta:
            return self.base_forward(x, l, l_mask, t_mask, p_mask)
        else:
            return self.tta_forward(x, l, l_mask, t_mask, p_mask)


    def init_text_enc(self, type):
        pretrained = 'pretrained_weights/bert/' + type

        if type in ['bert-base-uncased', 'bert-large-uncased']:
            text_encoder = BertModel.from_pretrained(pretrained)
            text_encoder.pooler = None
        else:
            raise NotImplementedError

        return text_encoder

    def language_forward(self, type, l, l_mask, prompts=None, t_mask=None, p_mask=None):
        # basic BERT
        if type in ['bert-base-uncased', 'bert-large-uncased']:
            l_feats = self.text_encoder(l, attention_mask=l_mask, vis_token=prompts)[0]  # (B, 20, 768)
            l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
            l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
            if self.enc_cfg['kwargs']['vlf_ris'] == 'FIANet':
                t_feats = self.text_encoder(l, attention_mask=t_mask)[0]
                t_feats = t_feats.permute(0, 2, 1)  # (B, 768, N_l)
                t_mask = t_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
                p_feats = self.text_encoder(l, attention_mask=p_mask)[0]
                p_feats = p_feats.permute(0, 2, 1)  # (B, 768, N_l)
                p_mask = p_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        else:
            raise NotImplementedError

        return l_feats, l_mask, None, None, None, None

    def various_decoder(self, type, feats, l_feats, l_mask, h, w):

        out = F.interpolate(self.decoder(feats, l_feats, l_mask), size=(h, w), mode='bilinear', align_corners=False)
        return out
