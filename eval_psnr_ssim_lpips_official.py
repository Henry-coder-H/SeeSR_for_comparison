#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高速评测 PSNR / SSIM / LPIPS（对齐 DIV2K 常用评测）
- 文件名自动配对（剥离 SR 名称末尾的 x4/x1.2/x1p2/×4/-sr 等后缀）
- 支持 Y 通道 (--y_only) 与边界裁剪 (--crop)，以及尺寸不一致时将 SR 双三次缩放到 GT (--resize_to_gt)
- 逐图打印结果，末尾打印平均值
- 用法示例：CUDA_VISIBLE_DEVICES=3 python /data4/huangsiyu/SeeSR/eval_psnr_ssim_lpips_official.py   --pred_dir ./experience/eval_div2k_x4/sample00   --gt_dir /data_center/data1/dataset/DIV2K/valid/valid_HR   --y_only  --crop 4  --lpips  --lpips_net alex
"""

import os, re, glob, math, argparse
import numpy as np
import cv2
from PIL import Image
import torch

# 可选 LPIPS（pip install lpips）
try:
    import lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


# ---------- 工具：读取、配对、转换 ----------

def imread_bgr(path: str) -> np.ndarray:
    """读取 BGR（OpenCV 直读），保留 uint8；自动去 alpha。"""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im  # HWC, BGR, uint8


def crop_border(img: np.ndarray, c: int) -> np.ndarray:
    if c <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 2*c or w <= 2*c:
        raise ValueError(f"裁边像素过大：crop={c}, 图像尺寸={w}x{h}")
    return img[c:h-c, c:w-c, ...]


# MATLAB 口径的 Y（与 BasicSR/EDSR 等一致），输入 BGR，输出单通道 0..255（范围约 16..235）
def bgr2y_matlab(bgr: np.ndarray) -> np.ndarray:
    b = bgr[..., 0].astype(np.float64)
    g = bgr[..., 1].astype(np.float64)
    r = bgr[..., 2].astype(np.float64)
    y = 16.0 + (24.966 * b + 128.553 * g + 65.481 * r) / 255.0
    return y  # HxW, float64


# ---------- 指标：PSNR / SSIM（高速） ----------

def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR，输入 uint8 或 float64（0..255），返回 dB。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_gray_255(a: np.ndarray, b: np.ndarray) -> float:
    """
    单通道 SSIM，0..255，11x11 高斯核，sigma=1.5（与常见 SR 评测一致）
    用 OpenCV 的 GaussianBlur 加速。
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k = 11
    s = 1.5
    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    mu1 = cv2.GaussianBlur(a, (k, k), s)
    mu2 = cv2.GaussianBlur(b, (k, k), s)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(a * a, (k, k), s) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (k, k), s) - mu2_sq
    sigma12   = cv2.GaussianBlur(a * b, (k, k), s) - mu12

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    return float(ssim_map.mean())


def ssim_bgr_255(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    """三通道 BGR 的 SSIM：三个通道灰度 SSIM 的平均。"""
    ch = []
    for c in range(3):
        ch.append(ssim_gray_255(a_bgr[..., c], b_bgr[..., c]))
    return float(np.mean(ch))


# ---------- 文件名配对（剥离后缀） ----------

_SUFFIX_PATTERNS = [
    r'([_\-\s]*x\d+(?:\.\d+)?)$',      # x4, x1.2
    r'([_\-\s]*x\d+p\d+)$',            # x1p2
    r'([_\-\s]*×\d+(?:\.\d+)?)$',      # ×4, ×1.2（乘号）
    r'([_\-\s]*sr)$',                  # -sr / _sr / sr
]

def normalize_key(stem: str) -> str:
    s = stem.lower()
    changed = True
    while changed:
        changed = False
        for pat in _SUFFIX_PATTERNS:
            s2 = re.sub(pat, '', s, flags=re.IGNORECASE)
            if s2 != s:
                s = s2
                changed = True
    return s


# ---------- LPIPS ----------

def to_lpips_tensor_from_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    """
    HWC BGR uint8 -> NCHW RGB float in [-1,1]
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    t = t * 2.0 - 1.0
    return t.unsqueeze(0)  # 1x3xHxW


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="推理结果目录（SR 输出）")
    ap.add_argument("--gt_dir",   required=True, help="GT 目录（DIV2K valid/HR 等）")
    ap.add_argument("--y_only", action="store_true", help="只在 Y 通道上评测（DIV2K 常用口径，建议配合 --crop）")
    ap.add_argument("--crop", type=int, default=0, help="评测时四边裁剪像素（x4 通常设 4；连续倍率建议 0）")
    ap.add_argument("--resize_to_gt", action="store_true", help="尺寸不一致时，将 SR 双三次缩放到 GT 尺寸后评测")
    ap.add_argument("--ext", type=str, default="png,jpg,jpeg", help="匹配的文件后缀（逗号分隔）")
    ap.add_argument("--lpips", action="store_true", help="计算 LPIPS（需 pip install lpips）")
    ap.add_argument("--lpips_net", type=str, default="alex", choices=["alex","vgg","squeeze"], help="LPIPS 主干网络")
    args = ap.parse_args()

    exts = tuple("." + e.strip().lower().lstrip(".") for e in args.ext.split(","))

    # 收集预测与 GT，建立“配对键 -> 路径”索引
    def stem(p): return os.path.splitext(os.path.basename(p))[0]

    pred_list = sorted([p for p in glob.glob(os.path.join(args.pred_dir, "*")) if p.lower().endswith(exts)])
    if not pred_list:
        raise FileNotFoundError(f"在 {args.pred_dir} 没找到图像（扩展名 {exts}）")

    gt_map = {}
    for p in glob.glob(os.path.join(args.gt_dir, "*")):
        if p.lower().endswith(exts):
            gt_map[normalize_key(stem(p))] = p

    # LPIPS 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = None
    if args.lpips:
        if not _HAS_LPIPS:
            raise RuntimeError("未安装 lpips：请先 `pip install lpips` 再使用 --lpips")
        loss_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    # 逐图评测
    sum_psnr, sum_ssim, sum_lp, n = 0.0, 0.0, 0.0, 0
    missed = []

    for p_pred in pred_list:
        sr_stem = stem(p_pred)
        key = normalize_key(sr_stem)
        p_gt = gt_map.get(key, None)
        if p_gt is None:
            missed.append(sr_stem)
            continue

        # 读取（BGR, uint8）
        sr = imread_bgr(p_pred)
        gt = imread_bgr(p_gt)

        # 尺寸对齐
        if sr.shape[:2] != gt.shape[:2]:
            if args.resize_to_gt:
                sr = cv2.resize(sr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                raise ValueError(f"尺寸不一致：{sr_stem} SR{sr.shape[:2]} vs GT{gt.shape[:2]}（可加 --resize_to_gt）")

        # 裁边
        sr_c = crop_border(sr, args.crop)
        gt_c = crop_border(gt, args.crop)

        # 选择评测通道
        if args.y_only:
            sr_eval = bgr2y_matlab(sr_c)  # HxW float64
            gt_eval = bgr2y_matlab(gt_c)
            cur_psnr = psnr_uint8(sr_eval, gt_eval)
            cur_ssim = ssim_gray_255(sr_eval, gt_eval)
        else:
            # 直接在 BGR 三通道上评测（与很多实现一致）
            cur_psnr = psnr_uint8(sr_c, gt_c)
            cur_ssim = ssim_bgr_255(sr_c, gt_c)

        # LPIPS（RGB [-1,1]）
        if args.lpips:
            with torch.no_grad():
                sr_t = to_lpips_tensor_from_bgr(sr_c).to(device)
                gt_t = to_lpips_tensor_from_bgr(gt_c).to(device)
                cur_lp = float(loss_fn(sr_t, gt_t).item())
        else:
            cur_lp = float("nan")

        sum_psnr += cur_psnr
        sum_ssim += cur_ssim
        if args.lpips:
            sum_lp += cur_lp
        n += 1

        if args.lpips:
            print(f"{sr_stem}: PSNR={cur_psnr:.4f} dB  SSIM={cur_ssim:.6f}  LPIPS={cur_lp:.6f}")
        else:
            print(f"{sr_stem}: PSNR={cur_psnr:.4f} dB  SSIM={cur_ssim:.6f}")

    if n == 0:
        raise RuntimeError("没有成功匹配到任何（SR, GT）对。请检查目录与命名或后缀剥离规则。")

    if args.lpips:
        print(f"平均: PSNR={sum_psnr/n:.4f} dB  SSIM={sum_ssim/n:.6f}  LPIPS={sum_lp/n:.6f}")
    else:
        print(f"平均: PSNR={sum_psnr/n:.4f} dB  SSIM={sum_ssim/n:.6f}")

    if missed:
        print(f"\n未找到 GT 的文件（已尝试去掉 x4/x3/x1p2/×4/-sr 后缀）：{len(missed)} 张，例如：{missed[:5]}")
    

if __name__ == "__main__":
    main()
