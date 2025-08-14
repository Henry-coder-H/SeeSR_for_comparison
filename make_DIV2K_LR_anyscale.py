#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 DIV2K 官方管线（Matlab 'bicubic' + antialiasing）从 HR 生成 LR_bicubic_X{scale}
- 优先使用 BasicSR 的 matlab_imresize（与 Matlab 几乎一致）
- 若环境里没有 BasicSR，则回退到 PIL.Image.BICUBIC（数值会有细微差别，不建议做严格对齐评测时使用）

用法举例（生成 1.2× 的 LR）：
python make_DIV2K_LR_anyscale.py \
  --hr_dir /data_center/data1/dataset/DIV2K/valid/valid_HR \
  --out_root /data_center/data1/dataset/DIV2K/valid \
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, required=True,
                        help="DIV2K 的 HR 目录（如 valid/valid_HR 或 train/train_HR）")
    parser.add_argument("--out_root", type=str, required=True,
                        help="输出根目录（通常是 DIV2K/{train,valid} 目录）")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"],
                        help="用于输出目录命名：{split}_LR_bicubic_X{scale}")
    parser.add_argument("--scales", type=float, nargs="+", required=True,
                        help="生成的放大倍率，如 1.2 2 3 4 等（下采样比例为 1/scale）")
    parser.add_argument("--ext", type=str, default="png", help="保存格式（默认 png）")
    args = parser.parse_args()

    # 收集 HR
    hr_list = sorted(
        glob.glob(os.path.join(args.hr_dir, "*.png")) +
        glob.glob(os.path.join(args.hr_dir, "*.jpg")) +
        glob.glob(os.path.join(args.hr_dir, "*.jpeg"))
    )
    if len(hr_list) == 0:
        raise FileNotFoundError(f"未在 {args.hr_dir} 找到 HR 图像")

    print(f"Found {len(hr_list)} HR images in {args.hr_dir}")
    if HAS_MATLAB_IMRESIZE:
        print("[Info] 使用 BasicSR matlab_imresize（与官方 DIV2K bicubic 一致）")
    else:
        print("[Warn] 未发现 BasicSR，回退到 PIL.BICUBIC：可能与官方 LR 存在轻微数值差异")

    os.makedirs(args.out_root, exist_ok=True)

    for s in args.scales:
        assert s > 1.0, f"scale={s} 必须 > 1"
        # 与 DIV2K 命名风格对齐
        folder = f"{args.split}_LR_bicubic_X{s}"
        out_dir = os.path.join(args.out_root, folder)
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
    print("你可以用 --image_path 指向新生成的 LR 目录，并把 --upscale 设为相同的倍率进行验证。")


if __name__ == "__main__":
    main()
