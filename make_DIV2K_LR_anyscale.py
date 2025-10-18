#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成低分辨率图像用于SeeSR推理
- 按 DIV2K 官方管线（Matlab 'bicubic' + antialiasing）从 HR 生成 LR_bicubic_X{scale}
- 优先使用 BasicSR 的 matlab_imresize（与 Matlab 几乎一致）
- 若环境里没有 BasicSR，则回退到 PIL.Image.BICUBIC（数值会有细微差别，不建议做严格对齐评测时使用）
- 支持任意输入图像和多种放大倍率

用法举例：
# 生成单个倍率的LR图像
python make_DIV2K_LR_anyscale.py \
  --input_dir /path/to/your/images \
  --output_dir /path/to/output \
  --scales 4.0

# 生成多个倍率的LR图像
python make_DIV2K_LR_anyscale.py \
  --input_dir /path/to/your/images \
  --output_dir /path/to/output \
  --scales 1.2 2.0 4.0

# 从DIV2K数据集生成（保持原有功能）
python make_DIV2K_LR_anyscale.py \
  --hr_dir /data_center/data1/dataset/DIV2K/valid/valid_HR \
  --out_root /data4/huangsiyu/SeeSR_baseline/LR_image_from_DIV2K \
  --split valid \
  --scales 1.2
"""
import os
import glob
import argparse
from PIL import Image
import numpy as np

# --- 尝试使用 BasicSR 的 matlab_imresize（推荐） ---
HAS_MATLAB_IMRESIZE = False
try:
    from basicsr.utils.matlab_functions import imresize as matlab_imresize
    HAS_MATLAB_IMRESIZE = True
except Exception:
    HAS_MATLAB_IMRESIZE = False


def to_uint8(img_float01: np.ndarray) -> np.ndarray:
    """float32 [0,1] -> uint8 [0,255]（四舍五入 + 裁剪）"""
    return np.clip(np.round(img_float01 * 255.0), 0, 255).astype(np.uint8)


def matlab_bicubic_downsample(pil_img: Image.Image, scale: float) -> Image.Image:
    """
    用 Matlab-bicubic（带 AA）把图片缩到 1/scale 尺寸。
    """
    assert scale > 1.0, "scale 是放大倍率（>1），这里会按 1/scale 下采样"
    if HAS_MATLAB_IMRESIZE:
        # matlab_imresize 接受 float32 [0,1]
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        # 直接传缩放因子（内部会做边界/抗锯齿/通道处理）
        out = matlab_imresize(arr, scale=1.0 / float(scale))
        return Image.fromarray(to_uint8(out))
    else:
        # 回退（数值与官方有差异，谨慎用于严格评测）
        new_w = max(1, int(round(pil_img.width  / float(scale))))
        new_h = max(1, int(round(pil_img.height / float(scale))))
        return pil_img.resize((new_w, new_h), Image.BICUBIC)


def main():
    parser = argparse.ArgumentParser(description="生成低分辨率图像用于SeeSR推理")
    
    # 输入参数组（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=str,
                            help="输入图像目录（支持任意图像）")
    input_group.add_argument("--hr_dir", type=str,
                            help="DIV2K 的 HR 目录（如 valid/valid_HR 或 train/train_HR）")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str,
                        help="输出目录（用于 --input_dir 模式）")
    parser.add_argument("--out_root", type=str,
                        help="输出根目录（用于 --hr_dir 模式，通常是 DIV2K/{train,valid} 目录）")
    
    # 其他参数
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"],
                        help="用于输出目录命名：{split}_LR_bicubic_X{scale}（仅用于 --hr_dir 模式）")
    parser.add_argument("--scales", type=float, nargs="+", required=True,
                        help="生成的放大倍率，如 1.2 2 3 4 等（下采样比例为 1/scale）")
    parser.add_argument("--ext", type=str, default="png", help="保存格式（默认 png）")
    parser.add_argument("--prefix", type=str, default="LR",
                        help="输出文件夹前缀（用于 --input_dir 模式，默认 'LR'）")
    
    args = parser.parse_args()

    # 验证参数
    if args.input_dir and not args.output_dir:
        parser.error("使用 --input_dir 时必须指定 --output_dir")
    if args.hr_dir and not args.out_root:
        parser.error("使用 --hr_dir 时必须指定 --out_root")

    # 确定输入目录和输出根目录
    if args.input_dir:
        input_dir = args.input_dir
        output_root = args.output_dir
        mode = "custom"
    else:
        input_dir = args.hr_dir
        output_root = args.out_root
        mode = "div2k"

    # 收集输入图像
    hr_list = sorted(
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.jpeg"))
    )
    if len(hr_list) == 0:
        raise FileNotFoundError(f"未在 {input_dir} 找到图像")

    print(f"Found {len(hr_list)} images in {input_dir}")
    print(f"Mode: {'Custom images' if mode == 'custom' else 'DIV2K dataset'}")
    if HAS_MATLAB_IMRESIZE:
        print("[Info] 使用 BasicSR matlab_imresize（与官方 DIV2K bicubic 一致）")
    else:
        print("[Warn] 未发现 BasicSR，回退到 PIL.BICUBIC：可能与官方 LR 存在轻微数值差异")

    os.makedirs(output_root, exist_ok=True)

    for s in args.scales:
        assert s > 1.0, f"scale={s} 必须 > 1"
        
        # 根据模式确定输出目录名称
        if mode == "custom":
            folder = f"{args.prefix}_X{s}"
        else:
            folder = f"{args.split}_LR_bicubic_X{s}"
            
        out_dir = os.path.join(output_root, folder)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n==> 生成 {folder} 到 {out_dir}")

        for i, hr_path in enumerate(hr_list, 1):
            name = os.path.splitext(os.path.basename(hr_path))[0]
            img_hr = Image.open(hr_path).convert("RGB")
            img_lr = matlab_bicubic_downsample(img_hr, s)
            save_path = os.path.join(out_dir, f"{name}.{args.ext}")
            img_lr.save(save_path)
            if i % 50 == 0 or i == len(hr_list):
                print(f"  [{i:4d}/{len(hr_list)}] {name} -> {folder}")

    print("\nDone.")
    if mode == "custom":
        print(f"生成的LR图像保存在: {output_root}")
        print("你可以用 --image_path 指向新生成的 LR 目录，并把 --upscale 设为相同的倍率进行推理。")
    else:
        print("你可以用 --image_path 指向新生成的 LR 目录，并把 --upscale 设为相同的倍率进行验证。")


if __name__ == "__main__":
    main()
