# LSCF
The code of paper "LSCF: Long-term Senmantic-guidance ConvFormer for Referring Image Segmentation".

## [LSCF: Long-Term Semantic-Guidance ConvFormer for Referring Remote Sensing Image Segmentation](https://ieeexplore.ieee.org/document/11029253)
Referring remote sensing image segmentation (RRSIS) task aims to generate segmentation masks for target objects based on language descriptions. It requires precise localization while distinguishing between visually similar yet semantically distinct objects. Fusing vision-language features only during extraction causes information loss and semantic forgetting in the decoder, harming similar target distinction. In addition, high-resolution remote sensing images present challenges, including complex backgrounds, diverse object scales, and intricate boundaries, limiting the effectiveness of previous methods. To address these issues, we propose the long-term semantic-guidance ConvFormer (LSCF) Network. First, we fuse multireceptive-field local features extracted by the multiscale coordconv (MCC) module with language-aware global features from the cross-modal attention (CA) module to obtain multimodal representations. Second, the sampling attention (SA) module enables fine-grained vision context alignment under semantic guidance. Finally, the global language fusion (GLF) module is incorporated in the decoder to maintain long-term vision-language alignment and mitigate semantic degradation. Experimental validation on the RefSegRS, RRSIS-D, and RISBench datasets demonstrates that LSCF achieves overall intersection of union (oIoU) scores of 83.27%, 77.42%, and 74.88%, and mean intersection over union (mIoU) scores of 77.44%, 64.25%, and 68.53%, respectively. On RefSegRS, LSCF surpasses the SOTA method FIANet by 5.53% (oIoU) and 9.58% (mIoU), while delivering competitive performance on RRSIS-D and RISBench.
![Overall Framework of LSCF](LSCF.png)

## 🌟🌟🌟
Our group build a unified deep learning framework, RSFM, dedicated to achieving remote sensing perception, generation, and interpretation in one library. RSFM splits and regularizes various scientific tasks into an intuitive and streamlined structure, aiming to provide researchers with a readable, easily editable, and beginner-friendly codebase. Within RSFM, researchers can freely mix and match model components from different tasks to explore their full potential. Although RSFM focuses on remote sensing data, it is not limited to this domain; it also seamlessly supports natural, medical, synthetic, and industrial data as well.
- Code: https://github.com/IPIU-XDU/RSFM
- Contributors: [Xiaoqiang Lu](https://github.com/xiaoqiang-lu), [Qin Ma](https://github.com/chunbai1), [Jiamin Cao](https://github.com/JMcarrot), [Jing Zhang](https://github.com/Jerry-jing), [Xinyu Liu](https://github.com/xxxxyliu), [Chenyue Che](https://github.com/chenyueche), [Yanyan Zu](https://github.com/Zuyanyan), [Yanzhao Zhang](https://github.com/stuzyz), [Jinming Chai](https://github.com/JMcarrot), [Long Sun](https://github.com/JMcarrot)
- Copyright (c) IPIU-XDU. All rights reserved.

## Installation
```shell 
conda create -n lscf python=3.10 -y
conda activate lscf
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install scikit-image transformers pycocotools
pip install tokenizers h5py

# other
apt install libgl1-mesa-glx
```

## Usage
### Training
```shell
bash scripts/train_dist.sh <num gpus> <port>
# for example: bash scripts/train_dist.sh 4 10000
```

### Validation
```shell
# Normal validation
bash scripts/eval_dist.sh <num gpus> <port> --weight-path <path/to/your/trained/weight>

# Using test-time augmentation (TTA), including multi-scale (x1.0, x1.125, x1.25, x1.375, x1.5) augs with horizontal flipping
bash scripts/eval_dist.sh <num gpus> <port> --weight-path <path/to/your/trained/weight> --tta
```

### Prediction
```shell
# Normal prediction
bash scripts/eval_dist.sh <num gpus> <port> --task predict --weight-path <path/to/your/trained/weight> --save-path <path/to/dir/you/want/save>

# Using test-time augmentation (TTA)
bash scripts/eval_dist.sh <num gpus> <port> --task predict --weight-path <path/to/your/trained/weight> --save-path <path/to/dir/you/want/save> --tta
```
