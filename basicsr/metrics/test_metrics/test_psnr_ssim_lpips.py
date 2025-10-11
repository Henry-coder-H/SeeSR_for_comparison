'''
用法示例（新增 LPIPS）：
cd /data4/huangsiyu/SeeSR
CUDA_VISIBLE_DEVICES=0 \
python -m basicsr.metrics.test_metrics.test_psnr_ssim_lpips \
  --sr_dir /data4/huangsiyu/SeeSR/experience/eval_div2k_x4/sample00 \
  --gt_dir  /data_center/data1/dataset/DIV2K/valid/valid_HR \
  --crop_border 4 \
  --y_only \
  --lpips_net alex
'''

import os
import re
import glob
import argparse
import cv2
import numpy as np
import torch

# basicsr 的 PSNR/SSIM（与官方 DIV2K 脚本一致）
from basicsr.metrics import calculate_psnr, calculate_ssim

# LPIPS
try:
    import lpips  # pip install lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


# ---------------- 工具函数 ----------------
def imread_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:  # 去 alpha
        img = img[:, :, :3]
    if img.ndim == 3:  # BGR->RGB（basicsr 的 Y 通道实现通常按 RGB）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  # HWC, uint8


# 去尾缀（支持多轮，处理 xxx_x4_sr 等）
_SUFFIX_PATTERNS = [
    r'([_\-\s]*x\d+(?:\.\d+)?)$',      # x4, x1.2
    r'([_\-\s]*x\d+p\d+)$',            # x1p2
    r'([_\-\s]*×\d+(?:\.\d+)?)$',      # ×4, ×1.2（乘号）
    r'([_\-\s]*sr)$',                  # -sr/_sr/sr
]
def normalize_key_from_stem(stem: str) -> str:
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


def build_index(dir_path, exts):
    """返回 {配对键: 路径}"""
    idx = {}
    for p in glob.glob(os.path.join(dir_path, "*")):
        if not os.path.isfile(p):
            continue
        if not p.lower().endswith(exts):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        key = normalize_key_from_stem(stem)
        idx[key] = p
    return idx


def to_lpips_tensor(img_np: np.ndarray, device: torch.device):
    """
    HWC, uint8 [0,255] -> NCHW, float32 in [-1,1]
    LPIPS 期望 3 通道 RGB。
    """
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=2)
    if img_np.shape[2] == 1:
        img_np = np.concatenate([img_np]*3, axis=2)

    t = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).to(device)  # [1,3,H,W]


# ---------------- 数据集评测 ----------------
def eval_dataset(sr_dir,
                 gt_dir,
                 crop_border=0,
                 test_y_channel=False,
                 resize_to_gt=False,
                 ext_list=("png","jpg","jpeg"),
                 lpips_net="alex",
                 lpips_cpu=False):
    exts = tuple(["." + e.lower().strip(". ") for e in ext_list])

    # 建 GT 索引：{配对键: GT路径}
    gt_index = build_index(gt_dir, exts)

    # 遍历 SR，按“配对键”找 GT
    sr_paths = sorted([p for p in glob.glob(os.path.join(sr_dir, "*")) if p.lower().endswith(exts)])
    if not sr_paths:
        raise FileNotFoundError(f"在 {sr_dir} 下没有找到图像（扩展名：{exts}）")

    # LPIPS 准备
    do_lpips = _HAS_LPIPS
    device = torch.device("cpu" if lpips_cpu or (not torch.cuda.is_available()) else "cuda")
    if do_lpips:
        loss_fn = lpips.LPIPS(net=lpips_net).to(device).eval()

    ps_list, ss_list, lp_list = [], [], []
    missed = []

    for p_sr in sr_paths:
        stem = os.path.splitext(os.path.basename(p_sr))[0]
        key  = normalize_key_from_stem(stem)
        p_gt = gt_index.get(key)

        if p_gt is None:
            missed.append(stem)
            continue

        sr = imread_rgb(p_sr)
        gt = imread_rgb(p_gt)

        # 尺寸不一致时的处理
        if sr.shape[:2] != gt.shape[:2]:
            if resize_to_gt:
                sr = cv2.resize(sr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                raise ValueError(f"尺寸不一致：{stem}  SR{sr.shape[:2]} vs GT{gt.shape[:2]}（可加 --resize_to_gt）")

        # ----- PSNR / SSIM（basicsr 的 numpy 实现；函数内部完成裁边 & Y 通道）
        cur_psnr = calculate_psnr(sr, gt, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
        cur_ssim = calculate_ssim(sr, gt, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)

        # ----- LPIPS：按 RGB 全通道计算；同样先裁边（若需要）
        cur_lp = None
        if do_lpips:
            if crop_border > 0:
                sr_lp = sr[crop_border:-crop_border, crop_border:-crop_border, ...]
                gt_lp = gt[crop_border:-crop_border, crop_border:-crop_border, ...]
            else:
                sr_lp, gt_lp = sr, gt

            with torch.no_grad():
                sr_t = to_lpips_tensor(sr_lp, device)
                gt_t = to_lpips_tensor(gt_lp, device)
                cur_lp = float(loss_fn(sr_t, gt_t).item())
            lp_list.append(cur_lp)

        if cur_lp is None:
            print(f"{stem}: PSNR={cur_psnr:.4f} dB, SSIM={cur_ssim:.6f}")
        else:
            print(f"{stem}: PSNR={cur_psnr:.4f} dB, SSIM={cur_ssim:.6f}, LPIPS={cur_lp:.6f}")

        ps_list.append(cur_psnr)
        ss_list.append(cur_ssim)

    if not ps_list:
        raise RuntimeError("没有成功匹配到任何 SR/GT 对。请检查目录与命名（或后缀剥离规则）。")

    if do_lpips and lp_list:
        print(f"平均: PSNR={np.mean(ps_list):.4f} dB, SSIM={np.mean(ss_list):.6f}, LPIPS={np.mean(lp_list):.6f}")
    else:
        print(f"平均: PSNR={np.mean(ps_list):.4f} dB, SSIM={np.mean(ss_list):.6f}")
        if not _HAS_LPIPS:
            print("[提示] 未安装 lpips，已跳过 LPIPS 计算。可执行：pip install lpips")

    if missed:
        print(f"\n未找到 GT 的文件（已自动尝试去掉 x4/x1.2/_sr 等尾缀）：{len(missed)} 张，例如：{missed[:5]}")


# ---------------- main（数据集入口） ----------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr_dir", required=True, help="SR 结果目录（你的推理输出）")
    ap.add_argument("--gt_dir", required=True, help="GT 目录")
    ap.add_argument("--crop_border", type=int, default=0, help="四边裁剪像素（DIV2K x4 常设 4；连续倍率常设 0）")
    ap.add_argument("--y_only", action="store_true", help="只在亮度(Y)通道评测（仅作用于 PSNR/SSIM）")
    ap.add_argument("--resize_to_gt", action="store_true", help="尺寸不一致时，把 SR 双三次缩放到 GT 尺寸再评")
    ap.add_argument("--ext", type=str, default="png,jpg,jpeg", help="后缀（逗号分隔）")

    # LPIPS 相关
    ap.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone")
    ap.add_argument("--lpips_cpu", action="store_true", help="在 CPU 上计算 LPIPS（默认自动用 GPU）")

    args = ap.parse_args()

    eval_dataset(
        sr_dir=args.sr_dir,
        gt_dir=args.gt_dir,
        crop_border=args.crop_border,
        test_y_channel=args.y_only,
        resize_to_gt=args.resize_to_gt,
        ext_list=[e.strip() for e in args.ext.split(",")],
        lpips_net=args.lpips_net,
        lpips_cpu=args.lpips_cpu,
    )
