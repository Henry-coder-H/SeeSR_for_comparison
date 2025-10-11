import glob
import os
import json
from PIL import Image
import random
import numpy as np
import re
import math

import torch
from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F

from .realesrgan import RealESRGAN_degradation

class PairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folders=None,
            tokenizer=None,
            null_text_ratio=0.5,
            use_log_scale=True,
            min_scale=1/16,
            max_scale=1.0,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(PairedCaptionDataset, self).__init__()

        self.null_text_ratio = null_text_ratio
        self.use_log_scale = use_log_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # Log-scale sampling parameters
        if self.use_log_scale:
            self.log_min_scale = math.log(min_scale)
            self.log_max_scale = math.log(max_scale)
        # self.lr_list = []
        # self.gt_list = []
        # self.tag_path_list = []

        # root_folders = root_folders.split(',')
        # for root_folder in root_folders:
        #     lr_path = root_folder +'/sr_bicubic'
        #     tag_path = root_folder +'/tag'
        #     gt_path = root_folder +'/gt'

        #     self.lr_list += glob.glob(os.path.join(lr_path, '*.png'))
        #     self.gt_list += glob.glob(os.path.join(gt_path, '*.png'))
        #     self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))


        # assert len(self.lr_list) == len(self.gt_list)
        # assert len(self.lr_list) == len(self.tag_path_list)

        self.lr_up_list = []        # sr_bicubic（作为条件图输入模型）
        self.gt_list = []           # GT
        self.tag_path_list = []     # 文本tag
        self.scale_list = []       # 直接来自 scale_meta.jsonl

        root_folders = root_folders.split(',')

        def _tag_key(p: str) -> str:
            """把 'tag0001234.txt' 规范化成 '0001234'；若文件名本来就是 '0001234.txt' 也OK。"""
            name = os.path.splitext(os.path.basename(p))[0]
            return re.sub(r"^tag", "", name)

        for root_folder in root_folders:
            sr_path  = os.path.join(root_folder, 'sr_bicubic')
            gt_path  = os.path.join(root_folder, 'gt')
            tag_dir  = os.path.join(root_folder, 'tag')
            meta_fp  = os.path.join(root_folder, 'scale_meta.jsonl')

            # tag 既支持 root/tag/*.txt 也支持 root/tag*.txt
            if os.path.isdir(tag_dir):
                tag_files = glob.glob(os.path.join(tag_dir, '*.txt'))
            else:
                tag_files = glob.glob(os.path.join(root_folder, 'tag*.txt'))

            sr_dict = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(sr_path, '*.png'))}
            gt_dict = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(gt_path, '*.png'))}
            tag_dict = {_tag_key(p): p for p in tag_files}

            # 读取 scale_meta.jsonl（关键）
            name2scale = {}
            if os.path.isfile(meta_fp):
                with open(meta_fp, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        # 兼容两种字段：{"id": "0000001", "scale": 2.0} 或 {"name": "0000001.png", "scale": 2.0}
                        if "id" in rec:
                            key = str(rec["id"])
                        elif "name" in rec:
                            key = os.path.splitext(os.path.basename(rec["name"]))[0]
                        else:
                            continue
                        name2scale[key] = float(rec["scale"])
            else:
                raise FileNotFoundError(f"缺少 {meta_fp}）")

            # 四者交集，确保严格一一对应
            common_ids = sorted(set(sr_dict) & set(gt_dict) & set(tag_dict) & set(name2scale))

            for k in common_ids:
                self.lr_up_list.append(sr_dict[k])
                self.gt_list.append(gt_dict[k])
                self.tag_path_list.append(tag_dict[k])
                self.scale_list.append(name2scale[k])

        print(f"匹配到的样本数量: {len(self.lr_up_list)}")
        assert len(self.lr_up_list) == len(self.gt_list) == len(self.tag_path_list) == len(self.scale_list)

        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])

        ram_mean = [0.485, 0.456, 0.406]
        ram_std  = [0.229, 0.224, 0.225]
        self.ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

        self.tokenizer = tokenizer

    def tokenize_caption(self, caption=""):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    # def __getitem__(self, index):

       
    #     gt_path = self.gt_list[index]
    #     gt_img = Image.open(gt_path).convert('RGB')
    #     gt_img = self.img_preproc(gt_img)
        
    #     lq_path = self.lr_list[index]
    #     lq_img = Image.open(lq_path).convert('RGB')
    #     lq_img = self.img_preproc(lq_img)

    #     if random.random() < self.null_text_ratio:
    #         tag = ''
    #     else:
    #         tag_path = self.tag_path_list[index]
    #         file = open(tag_path, 'r')
    #         tag = file.read()
    #         file.close()

    #     example = dict()
    #     example["conditioning_pixel_values"] = lq_img.squeeze(0)
    #     example["pixel_values"] = gt_img.squeeze(0) * 2.0 - 1.0
    #     example["input_ids"] = self.tokenize_caption(caption=tag).squeeze(0)

    #     lq_img = lq_img.squeeze()

    #     ram_values = F.interpolate(lq_img.unsqueeze(0), size=(384, 384), mode='bicubic')
    #     ram_values = ram_values.clamp(0.0, 1.0)
    #     example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))

    #     return example

    def __getitem__(self, index):
        # GT
        gt_path = self.gt_list[index]
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = self.img_preproc(gt_img)  # (C,H,W)

        # 条件图（已双三次对齐到 GT 尺寸）
        lq_up_path = self.lr_up_list[index]
        lq_up_img = Image.open(lq_up_path).convert('RGB')
        lq_up_img = self.img_preproc(lq_up_img)  # (C,H,W)

        # 标签到文本
        if random.random() < self.null_text_ratio:
            tag = ''
        else:
            tag_path = self.tag_path_list[index]
            with open(tag_path, 'r') as f:
                tag = f.read()

        # 关键：处理scale，现在meta文件中保存的就是log-scale值
        if self.use_log_scale:
            # 直接使用meta中的log-scale值
            scale = torch.tensor(self.scale_list[index], dtype=torch.float32)
        else:
            # 如果meta中保存的是原始scale值，转换为log-scale
            original_scale = self.scale_list[index]
            scale = torch.tensor(math.log(original_scale), dtype=torch.float32)

        example = dict()
        example["conditioning_pixel_values"] = lq_up_img                 # (C,H,W)
        example["pixel_values"] = gt_img * 2.0 - 1.0                     # [-1,1]
        example["input_ids"] = self.tokenize_caption(caption=tag).squeeze(0)

        # RAM 输入（基于条件图）
        ram_values = F.interpolate(lq_up_img.unsqueeze(0), size=(384, 384), mode='bicubic').clamp(0.0, 1.0)
        example["ram_values"] = self.ram_normalize(ram_values.squeeze(0))

        # 返回 scale（DataLoader 会组合成 (B,)）
        example["scale"] = scale

        return example

    def __len__(self):
        return len(self.gt_list)