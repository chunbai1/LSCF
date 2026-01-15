# Hyper-parameters setting for various types of backbone

encoder_info = {
    'swin_tiny': {'enc_out_dims': [96, 192, 384, 768]},
    'swin_small': {'enc_out_dims': [96, 192, 384, 768]},
    'swin_base': {'enc_out_dims': [128, 256, 512, 1024]},
    'swin_base_gfm_w6': {'enc_out_dims': [128, 256, 512, 1024]},
    'swin_large': {'enc_out_dims': [192, 384, 768, 1536]},
}

decoder_info = {
    'FreqLAVTHead': {'dec_out_dim': 384},
}

language_info = {
    'bert-base-uncased': {'hidden_dim': 768},
    'bert-large-uncased': {'hidden_dim': 1024},
}
