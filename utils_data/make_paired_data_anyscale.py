# make_paired_data_anyscale.py
"""
SeeSR 任意倍率数据构建（基于 RealESRGAN 退化 + 连续 scale）
输入：DIV2K 的 HR 目录
输出：out_root/{gt, sr_bicubic} + scale_meta.jsonl（记录每张图的倍率）
后续再跑打标签脚本得到 out_root/tag/*.txt

注意：此脚本在log空间采样scale，然后转换为原始scale值进行图像处理
log-scale采样范围：[log(1/16), 0] 对应原始scale范围：[1/16, 1.0]

生成代码：CUDA_VISIBLE_DEVICES=0 python /data4/huangsiyu/SeeSR_baseline/utils_data/make_paired_data_anyscale.py \
--hr_dir /data_center/data1/dataset/DIV2K/train/train_HR \
--out_root /data4/huangsiyu/SeeSR_baseline/preset/datasets/train_datasets/training_for_seesr \
--epoch 1
"""
import os, sys, cv2, math, random, argparse, json
sys.path.append(os.getcwd())
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from tqdm import tqdm

from basicsr.data.realesrgan_dataset import RealESRGANDataset
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

parser = argparse.ArgumentParser()
parser.add_argument("--hr_dir", type=str, required=True, help="DIV2K train_HR 目录")
parser.add_argument("--out_root", type=str, required=True, help="输出根目录，例如 preset/datasets/train_datasets/div2k")
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=2, help="smaller batch size means much time but more extensive degradation for making the training dataset.")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--gt_patch", type=int, default=256, help="GT patch size")
parser.add_argument("--log_scale_min", type=float, default=math.log(1/16), help="Minimum log-scale value (log(1/16) ≈ -2.77)")
parser.add_argument("--log_scale_max", type=float, default=0.0, help="Maximum log-scale value (log(1) = 0)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

seed_everything(args.seed)

# --------- RealESRGAN dataset config（和 SeeSR 一致，只调 patch 大小） ----------
cfg_ds = dict(
    gt_path=[args.hr_dir],
    queue_size=160,
    crop_size=args.gt_patch,
    io_backend={"type": "disk"},

    blur_kernel_size=21,
    kernel_list=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    kernel_prob=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    sinc_prob=0.1,
    blur_sigma=[0.2, 3],
    betag_range=[0.5, 4],
    betap_range=[1, 2],

    blur_kernel_size2=11,
    kernel_list2=['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    kernel_prob2=[0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    sinc_prob2=0.1,
    blur_sigma2=[0.2, 1.5],
    betag_range2=[0.5, 4.0],
    betap_range2=[1, 2],

    final_sinc_prob=0.8,
    use_hflip=True,
    use_rot=False,
)

# 添加no_degradation_prob参数（与原版保持一致）
cfg_ds['no_degradation_prob'] = 0.01
train_dataset = RealESRGANDataset(cfg_ds)
loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size,
                    num_workers=args.num_workers, drop_last=True)

# --------- degradation knobs（与原始一致） ----------
cfg_deg = {
    "resize_prob": [0.2, 0.7, 0.1],        # up, down, keep
    "resize_range": [0.15, 1.5],
    "gaussian_noise_prob": 0.5,
    "noise_range": [1, 30],
    "poisson_scale_range": [0.05, 3.0],
    "gray_noise_prob": 0.4,
    "jpeg_range": [30, 95],

    "second_blur_prob": 0.8,
    "resize_prob2": [0.3, 0.4, 0.3],
    "resize_range2": [0.3, 1.2],
    "gaussian_noise_prob2": 0.5,
    "noise_range2": [1, 25],
    "poisson_scale_range2": [0.05, 2.5],
    "gray_noise_prob2": 0.4,
    "jpeg_range2": [30, 95],

    "gt_size": args.gt_patch,
    "no_degradation_prob": 0.01,
}

# --------- output dirs ----------
gt_dir = os.path.join(args.out_root, "gt")
sr_bic_dir = os.path.join(args.out_root, "sr_bicubic")  # 训练里用这个当 LQ（已经是上采样回 GT 尺寸）
tag_dir = os.path.join(args.out_root, "tag")
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(sr_bic_dir, exist_ok=True)
os.makedirs(tag_dir, exist_ok=True)

# 元数据文件（存 scale）
meta_path = os.path.join(args.out_root, "scale_meta.jsonl")
# 若存在则清空，避免重复
open(meta_path, "w").close()

# --------- ops on cuda ----------
jpeger = DiffJPEG(differentiable=False).cuda()
usm  = USMSharp().cuda()

def realesrgan_degradation_anyscale(batch, cfg_deg, sf_any: float):
    """核心退化流程，基于原版实现但支持任意倍率"""
    jpeger = DiffJPEG(differentiable=False).cuda()
    usm_sharpener = USMSharp().cuda()  # do usm sharpening
    im_gt = batch['gt'].cuda()
    im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch['kernel1'].cuda()
    kernel2 = batch['kernel2'].cuda()
    sinc_kernel = batch['sinc_kernel'].cuda()

    ori_h, ori_w = im_gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(im_gt, kernel1)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            cfg_deg['resize_prob'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, cfg_deg['resize_range'][1])
    elif updown_type == 'down':
        scale = random.uniform(cfg_deg['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = cfg_deg['gray_noise_prob']
    if random.random() < cfg_deg['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=cfg_deg['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=cfg_deg['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg_deg['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if random.random() < cfg_deg['second_blur_prob']:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            cfg_deg['resize_prob2'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, cfg_deg['resize_range2'][1])
    elif updown_type == 'down':
        scale = random.uniform(cfg_deg['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
            out,
            size=(int(ori_h / sf_any * scale),
                    int(ori_w / sf_any * scale)),
            mode=mode,
            )
    # add noise
    gray_noise_prob = cfg_deg['gray_noise_prob2']
    if random.random() < cfg_deg['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=cfg_deg['noise_range2'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=cfg_deg['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
            )

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if random.random() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h // sf_any),
                        int(ori_w // sf_any)),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg_deg['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg_deg['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h // sf_any),
                        int(ori_w // sf_any)),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)

    # clamp and round
    im_lq = torch.clamp(out, 0, 1.0)
    im_gt = torch.clamp(im_gt, 0, 1)

    # 清理中间变量
    del out, kernel1, kernel2, sinc_kernel
    torch.cuda.empty_cache()

    return im_lq, im_gt
        

step = 0
with open(meta_path, "a") as meta_f:
    for ep in range(args.epoch):
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {ep+1}/{args.epoch}")):
            # 每处理几个batch就清理一次显存
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
            
            # 在log空间均匀采样
            log_scale = random.uniform(args.log_scale_min, args.log_scale_max)
            # 转换为原始scale值用于图像处理
            sf_any = math.exp(log_scale)
            lq_full, gt_full = realesrgan_degradation_anyscale(batch, cfg_deg, sf_any)

            # 为保存成训练对：从 LQ/GT 各裁一个随机 patch，并把 LQ 双三次上采回 GT 尺寸，作为 sr_bicubic
            B, _, H, W = gt_full.shape
            # LQ patch 尺寸约为 GT_patch / sf_any
            lq_patch_size = max(8, int(round(args.gt_patch / float(sf_any))))
            for i in range(B):
                step += 1
                name = f"{step:07d}"

                # 随机在 LQ 上取窗口
                lq_i = lq_full[i:i+1]
                gt_i = gt_full[i:i+1]
                h, w = lq_i.shape[-2:]
                top_lq = 0 if h - lq_patch_size <= 0 else random.randint(0, h - lq_patch_size)
                left_lq = 0 if w - lq_patch_size <= 0 else random.randint(0, w - lq_patch_size)
                lq_patch = lq_i[:, :, top_lq:top_lq+lq_patch_size, left_lq:left_lq+lq_patch_size]

                # 对应映射回 GT（取 center 对齐更稳，也可用 round(top*sf)）
                top_gt = int(round(top_lq * sf_any))
                left_gt = int(round(left_lq * sf_any))
                top_gt = min(max(0, top_gt), gt_i.shape[-2] - args.gt_patch)
                left_gt = min(max(0, left_gt), gt_i.shape[-1] - args.gt_patch)
                gt_patch = gt_i[:, :, top_gt:top_gt+args.gt_patch, left_gt:left_gt+args.gt_patch]

                # LQ 上采回 GT 尺寸，作为 sr_bicubic
                sr_bic = F.interpolate(lq_patch, size=(args.gt_patch, args.gt_patch), mode='bicubic', align_corners=False)

                # 保存
                gt_np = (gt_patch[0].detach().cpu().permute(1,2,0).numpy()*255.0).clip(0,255).astype('uint8')[:, :, ::-1]
                lq_np = (sr_bic[0].detach().cpu().permute(1,2,0).numpy()*255.0).clip(0,255).astype('uint8')[:, :, ::-1]
                cv2.imwrite(os.path.join(gt_dir, f"{name}.png"), gt_np)
                cv2.imwrite(os.path.join(sr_bic_dir, f"{name}.png"), lq_np)

                # 记录元数据（关键：保存log-scale值）
                meta_f.write(json.dumps({"id": name, "scale": float(log_scale)}) + "\n")
            
            del lq_full, gt_full
            torch.cuda.empty_cache()

# 数据生成完成，下一步运行打标脚本生成 tag/
