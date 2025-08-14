# import cv2
# import torch

# from basicsr.metrics import calculate_psnr, calculate_ssim
# from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
# from basicsr.utils import img2tensor


# def test(img_path, img_path2, crop_border, test_y_channel=False):
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread(img_path2, cv2.IMREAD_UNCHANGED)

#     # --------------------- Numpy ---------------------
#     psnr = calculate_psnr(img, img2, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
#     ssim = calculate_ssim(img, img2, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
#     print(f'\tNumpy\tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')

#     # --------------------- PyTorch (CPU) ---------------------
#     img = img2tensor(img / 255., bgr2rgb=True, float32=True).unsqueeze_(0)
#     img2 = img2tensor(img2 / 255., bgr2rgb=True, float32=True).unsqueeze_(0)

#     psnr_pth = calculate_psnr_pt(img, img2, crop_border=crop_border, test_y_channel=test_y_channel)
#     ssim_pth = calculate_ssim_pt(img, img2, crop_border=crop_border, test_y_channel=test_y_channel)
#     print(f'\tTensor (CPU) \tPSNR: {psnr_pth[0]:.6f} dB, \tSSIM: {ssim_pth[0]:.6f}')

#     # --------------------- PyTorch (GPU) ---------------------
#     img = img.cuda()
#     img2 = img2.cuda()
#     psnr_pth = calculate_psnr_pt(img, img2, crop_border=crop_border, test_y_channel=test_y_channel)
#     ssim_pth = calculate_ssim_pt(img, img2, crop_border=crop_border, test_y_channel=test_y_channel)
#     print(f'\tTensor (GPU) \tPSNR: {psnr_pth[0]:.6f} dB, \tSSIM: {ssim_pth[0]:.6f}')

#     psnr_pth = calculate_psnr_pt(
#         torch.repeat_interleave(img, 2, dim=0),
#         torch.repeat_interleave(img2, 2, dim=0),
#         crop_border=crop_border,
#         test_y_channel=test_y_channel)
#     ssim_pth = calculate_ssim_pt(
#         torch.repeat_interleave(img, 2, dim=0),
#         torch.repeat_interleave(img2, 2, dim=0),
#         crop_border=crop_border,
#         test_y_channel=test_y_channel)
#     print(f'\tTensor (GPU batch) \tPSNR: {psnr_pth[0]:.6f}, {psnr_pth[1]:.6f} dB,'
#           f'\tSSIM: {ssim_pth[0]:.6f}, {ssim_pth[1]:.6f}')


# if __name__ == '__main__':

#     test('tests/data/bic/baboon.png', 'tests/data/gt/baboon.png', crop_border=4, test_y_channel=False)
#     test('tests/data/bic/baboon.png', 'tests/data/gt/baboon.png', crop_border=4, test_y_channel=True)

#     test('tests/data/bic/comic.png', 'tests/data/gt/comic.png', crop_border=4, test_y_channel=False)
#     test('tests/data/bic/comic.png', 'tests/data/gt/comic.png', crop_border=4, test_y_channel=True)




''' 用法示例：CUDA_VISIBLE_DEVICES=0 python /data4/huangsiyu/SeeSR/basicsr/metrics/test_metrics/test_psnr_ssim.py --sr_dir /data4/huangsiyu/SeeSR/experience/eval_div2k_x4/sample00   --gt_dir /data_center/data1/dataset/DIV2K/valid/valid_HR   --crop_border 4 -
-y_only
'''
import os
import re
import glob
import argparse
import cv2
import numpy as np

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from basicsr.utils import img2tensor

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

# ---------------- 数据集评测 ----------------
def eval_dataset(sr_dir, gt_dir, crop_border=0, test_y_channel=False, resize_to_gt=False, ext_list=("png","jpg","jpeg")):
    exts = tuple(["." + e.lower().strip(". ") for e in ext_list])

    # 建 GT 索引：{配对键: GT路径}
    gt_index = build_index(gt_dir, exts)

    # 遍历 SR，按“配对键”找 GT
    sr_paths = sorted([p for p in glob.glob(os.path.join(sr_dir, "*")) if p.lower().endswith(exts)])
    if not sr_paths:
        raise FileNotFoundError(f"在 {sr_dir} 下没有找到图像（扩展名：{exts}）")

    ps_list, ss_list = [], []
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

        # 直接用 basicsr 的 numpy 版本；裁边与 Y 通道在函数内部处理
        cur_psnr = calculate_psnr(sr, gt, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)
        cur_ssim = calculate_ssim(sr, gt, crop_border=crop_border, input_order='HWC', test_y_channel=test_y_channel)

        print(f"{stem}: PSNR={cur_psnr:.4f} dB, SSIM={cur_ssim:.6f}")

        ps_list.append(cur_psnr)
        ss_list.append(cur_ssim)

    if not ps_list:
        raise RuntimeError("没有成功匹配到任何 SR/GT 对。请检查目录与命名（或后缀剥离规则）。")

    print(f"平均: PSNR={np.mean(ps_list):.4f} dB, SSIM={np.mean(ss_list):.6f}")

    if missed:
        print(f"\n未找到 GT 的文件（已自动尝试去掉 x4/x1.2/_sr 等尾缀）：{len(missed)} 张，例如：{missed[:5]}")

# ---------------- main（改为数据集入口） ----------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr_dir", required=True, help="SR 结果目录（你的推理输出）")
    ap.add_argument("--gt_dir", required=True, help="GT 目录")
    ap.add_argument("--crop_border", type=int, default=0, help="四边裁剪像素（DIV2K x4 常设 4；连续倍率常设 0）")
    ap.add_argument("--y_only", action="store_true", help="只在亮度(Y)通道评测")
    ap.add_argument("--resize_to_gt", action="store_true", help="尺寸不一致时，把 SR 双三次缩放到 GT 尺寸再评")
    ap.add_argument("--ext", type=str, default="png,jpg,jpeg", help="后缀（逗号分隔）")
    args = ap.parse_args()

    eval_dataset(
        sr_dir=args.sr_dir,
        gt_dir=args.gt_dir,
        crop_border=args.crop_border,
        test_y_channel=args.y_only,
        resize_to_gt=args.resize_to_gt,
        ext_list=[e.strip() for e in args.ext.split(",")]
    )

