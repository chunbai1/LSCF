from torch.utils.data import Dataset
from os.path import join as osp
from language import BertTokenizer
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from os.path import join as osp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

datasets_info = {
    'RRSIS-D':
        {'data_root': 'RSFM_RIS_Datasets/RRSIS-D',
         'classes_name':
             ['background', 'foreground'],
         'img_suffix': '.jpg',
         'mask_suffix': '.png',
         'num_bands': 3,  # number of channels of your input data
         'num_classes': 2,
         'color_map':
             [[255, 255, 255], [255, 0, 0]],},
    'RefSegRS':
        {'data_root': 'RSFM_RIS_Datasets/RefSegRS',
         'classes_name':
             ['background', 'foreground'],
         'img_suffix': '.tif',
         'mask_suffix': '.tif',
         'num_bands': 3,  # number of channels of your input data
         'num_classes': 2,
         'color_map':
             [[255, 255, 255], [255, 0, 0]],},
    'RISBench':
        {'data_root': 'RSFM_RIS_Datasets/RISBench',
         'classes_name':
             ['background', 'foreground'],
         'img_suffix': '.png',
         'mask_suffix': '.png',
         'num_bands': 3,  # number of channels of your input data
         'num_classes': 2,
         'color_map':
             [[255, 255, 255], [255, 0, 0]]}
}

def create_dataset(cfg, rank, logger):

    trainset, valset = create_ris_dataset(cfg, mode='train'), create_ris_dataset(cfg, mode='val')

    if rank == 0:
        logger.info('Finding {:} images for training'.format(len(trainset)))
        logger.info('Finding {:} images for validation'.format(len(valset)))

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)

    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], pin_memory=False,
                                num_workers=4, drop_last=True, sampler=trainsampler)
    valloader = DataLoader(valset, batch_size=1,
                               pin_memory=False, num_workers=4, drop_last=False, sampler=valsampler)
    # valloader = DataLoader(valset, batch_size=1,
    #                            pin_memory=False, num_workers=4, drop_last=False, shuffle=False)

    return trainloader, valloader, trainsampler, valset

class create_test_dataset(Dataset):
    def __init__(self, task, dataset, mode='test', infer_size=None, txt_path=False):
        self.task = task
        self.dataset = dataset
        self.mode = mode
        txt_path = txt_path
        self.size = infer_size

        dataset_info = datasets_info[self.dataset]
        cfg = {}
        cfg['dataset'] = dataset

        self.root = dataset_info.get('data_root')
        self.img_suffix = dataset_info.get('img_suffix')
        self.mask_suffix = dataset_info.get('mask_suffix')
        self.classes_dict = {cat: i for i, cat in enumerate(dataset_info.get('classes_name'))}

        txt_path = osp(self.root, 'phrase_txts', 'output_phrase_test.txt')
        
        self.max_tokens = 20
        self.tokenizer = BertTokenizer.from_pretrained('pretrained_weights/bert/bert-base-uncased',
                                                            do_lower_case=True)
        self.ids, self.sentences, self.input_ids, self.attention_masks = load_id_phrase_info(txt_path, self.max_tokens, self.tokenizer)

    def __getitem__(self, item):
        id = self.ids[item]
        
        img, mask = load_img_mask_file(img_dir=osp(self.root, 'images', id + self.img_suffix),
                                        mask_dir=osp(self.root, 'masks', id + self.mask_suffix),
                                        convert_255_1=True)
        img, mask = resize_fixed(img, mask, self.size)
        img, mask = normalize(img, mask)

        choice_sent = np.random.choice(len(self.input_ids[item]))
        tensor_embeddings = self.input_ids[item][choice_sent]
        attention_mask = self.attention_masks[item][choice_sent]

        if self.task == 'eval':
            return img, mask, tensor_embeddings, attention_mask
        else:
            return img, tensor_embeddings, attention_mask, id

    def __len__(self):
        return len(self.ids)

def base_init(cfg):
    dataset_info = datasets_info.get(cfg['dataset'])

    root = cfg.get('data_root', dataset_info.get('data_root'))
    size = cfg.get('crop_size', dataset_info.get('training_size'))
    img_suffix = cfg.get('img_suffix', dataset_info.get('img_suffix'))
    mask_suffix = cfg.get('mask_suffix', dataset_info.get('mask_suffix'))
    num_bands = cfg['model']['backbone']['kwargs']['in_channels']

    return root, size, img_suffix, mask_suffix, num_bands

class create_ris_dataset(Dataset):
    def __init__(self, cfg, mode='train', eval_mode=False):
        self.root, self.size, self.img_suffix, self.mask_suffix, self.num_bands = base_init(cfg)

        self.max_tokens = 20
        self.tokenizer = BertTokenizer.from_pretrained('pretrained_weights/bert/' + cfg['model']['text_encoder']['type'])

        self.mode = mode
        if mode == 'train':
            txt_path = osp(self.root, 'phrase_txts', 'output_phrase_train.txt')
        elif mode == 'val':
            txt_path = osp(self.root, 'phrase_txts', 'output_phrase_val.txt')
        else:
            raise NotImplementedError

        self.ids, self.sentences, self.input_ids, \
        self.attention_masks = load_id_phrase_info(txt_path, self.max_tokens, self.tokenizer)

        self.eval_mode = eval_mode

    def __getitem__(self, item):
        id = self.ids[item]

        ### data augmentation may result in misalignment between image-text pair
        img, mask = load_img_mask_file(img_dir=osp(self.root, 'images', id + self.img_suffix),
                                       mask_dir=osp(self.root, 'masks', id + self.mask_suffix),
                                       convert_255_1=True)
        img, mask = resize_fixed(img, mask, self.size)
        img, mask = normalize(img, mask, self.num_bands)

        if self.eval_mode:
            embedding = []
            att = []
            for s in range(len(self.input_ids[item])):
                e = self.input_ids[item][s]
                a = self.attention_masks[item][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.input_ids[item]))
            tensor_embeddings = self.input_ids[item][choice_sent]
            attention_mask = self.attention_masks[item][choice_sent]

        return img, mask, tensor_embeddings, attention_mask

    def __len__(self):
        return len(self.ids)

def load_id_phrase_info(txt_path, max_tokens, tokenizer):

    def _process(var):
        if isinstance(var, str):
            return [var]
        elif isinstance(var, list):
            return var
        else:
            raise TypeError("Text input must be str or list !")

    ids, sentences, input_ids, attention_masks = [], [], [], []

    if txt_path.endswith('.txt'):
        for line in open(txt_path, 'r').readlines():
            info = line.strip().split(' ')
            ids.append(info[0])
            sentences.append(' '.join(i for i in info[1:]))
    else:
        raise NotImplementedError

    for i in range(len(ids)):

        img_sentences = _process(sentences[i])

        sentences_for_ref = []
        attentions_for_ref = []

        for sentence_raw in img_sentences:
            attention_mask = [0] * max_tokens
            padded_input_id = [0] * max_tokens

            input_id = tokenizer.encode(text=sentence_raw, add_special_tokens=True)
            # truncation of tokens
            input_id = input_id[:max_tokens]

            padded_input_id[:len(input_id)] = input_id
            attention_mask[:len(input_id)] = [1]*len(input_id)

            sentences_for_ref.append(torch.tensor(padded_input_id).unsqueeze(0))
            attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

        input_ids.append(sentences_for_ref)
        attention_masks.append(attentions_for_ref)

    return ids, sentences, input_ids, attention_masks

def load_img_mask_file(img_dir, mask_dir,
                       need_mask=True, convert_255_1=False):
    
    img = Image.fromarray(cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB))

    if need_mask:
        mask = Image.fromarray(mask_preprocess(cv2.imread(mask_dir, 0), convert_255_1))
        return img, mask

    return img

def resize_fixed(img, mask, size):
    img = img.resize((size, size), Image.BILINEAR)
    if mask is not None:
        mask = mask.resize((size, size), Image.NEAREST)
        return img, mask

    return img

def normalize(img, mask=None, num_bands=3):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(extend_list([0.485, 0.456, 0.406], num_bands),
                             extend_list([0.229, 0.224, 0.225], num_bands)),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def extend_list(original_list, num_bands):
    extended_list = original_list.copy()
    original_len = len(original_list)
    if num_bands > original_len:
        extension = [original_list[i % original_len] for i in range(original_len, num_bands)]
        extended_list.extend(extension)
    return extended_list

def min_max_normalize(mask):
    normalized_image = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-10) * 255
    result = normalized_image.astype(np.uint8)
    return result

def mask_preprocess(mask, convert_255_1=False):
    if convert_255_1:
        # assert np.all(mask[mask != 0] == 255)
        mask[mask != 0] = 1
        return mask

    return mask