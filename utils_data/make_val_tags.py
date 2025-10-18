#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为DIV2K-Val验证集生成标签文件
基于make_tags_anyscale.py的逻辑，为验证数据集生成RAM标签
"""

import os
import sys
import glob
import argparse
sys.path.append(os.getcwd())
import torch
from PIL import Image
from torchvision import transforms
from ram.models.ram import ram
from ram import inference_ram as inference

# RAM模型预处理
ram_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def main():
    parser = argparse.ArgumentParser(description="为DIV2K-Val验证集生成标签文件")
    parser.add_argument("--val_root", type=str, required=True,
                        help="DIV2K-Val验证集根目录，包含gt/和sr_bicubic/子目录")
    parser.add_argument("--start_gpu", type=int, default=0,
                        help="起始GPU索引（用于多GPU并行处理）")
    parser.add_argument("--all_gpu", type=int, default=1,
                        help="总GPU数量（用于多GPU并行处理）")
    parser.add_argument("--ram_model_path", type=str, 
                        default="preset/models/ram_swin_large_14m.pth",
                        help="RAM模型路径")
    
    args = parser.parse_args()
    
    # 设置路径
    gt_path = os.path.join(args.val_root, 'gt')
    tag_path = os.path.join(args.val_root, 'tag')
    
    # 检查GT目录是否存在
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"GT目录不存在: {gt_path}")
    
    # 创建标签目录
    os.makedirs(tag_path, exist_ok=True)
    
    # 获取GT图像列表
    gt_lists = sorted(glob.glob(os.path.join(gt_path, '*.png')))
    print(f'找到 {len(gt_lists)} 张GT图像')
    
    if len(gt_lists) == 0:
        print("错误: 未找到GT图像")
        return
    
    # 加载RAM模型
    print(f"加载RAM模型: {args.ram_model_path}")
    model = ram(pretrained=args.ram_model_path,
                image_size=384, vit='swin_l').eval().to('cuda')
    
    # 计算处理范围（支持多GPU并行）
    start = args.start_gpu * len(gt_lists) // args.all_gpu
    end = (args.start_gpu + 1) * len(gt_lists) // args.all_gpu
    print(f'处理范围: [{start}, {end}) (总共 {len(gt_lists)} 张图像)')
    
    # 生成标签
    with torch.no_grad():
        for i, gt_file in enumerate(gt_lists[start:end], start=start):
            basename = os.path.splitext(os.path.basename(gt_file))[0]  # 0000001
            
            # 加载并预处理图像
            img = ram_transforms(Image.open(gt_file).convert("RGB")).unsqueeze(0).cuda()
            
            # 生成标签
            captions = inference(img, model)
            prompt = f"{captions[0]},"
            
            # 保存标签文件
            save_txt = os.path.join(tag_path, f"{basename}.txt")
            with open(save_txt, "w") as f:
                f.write(prompt)
            
            # 打印进度
            if (i - start) % 100 == 0:
                print(f"[{i}/{end}] 已处理: {save_txt}")
    
    print(f"标签生成完成！")
    print(f"标签文件保存在: {tag_path}")
    print(f"总共生成了 {end - start} 个标签文件")

if __name__ == "__main__":
    main()
