#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIV2K-Val验证集评估脚本
计算SeeSR推理结果与GT图像的PSNR、SSIM、LPIPS等指标
"""

import os
import argparse
import glob
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 尝试导入评估指标库
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    print("警告: 未安装scikit-image，将跳过PSNR和SSIM计算")
    HAS_SKIMAGE = False

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    print("警告: 未安装lpips，将跳过LPIPS计算")
    HAS_LPIPS = False

def load_image(path):
    """加载图像并转换为numpy数组"""
    img = Image.open(path).convert('RGB')
    return np.array(img)

def calculate_psnr(img1, img2):
    """计算PSNR"""
    if not HAS_SKIMAGE:
        return 0.0
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    """计算SSIM"""
    if not HAS_SKIMAGE:
        return 0.0
    return ssim(img1, img2, data_range=255, multichannel=True, channel_axis=2)

def calculate_lpips(img1, img2, lpips_model):
    """计算LPIPS"""
    if not HAS_LPIPS or lpips_model is None:
        return 0.0
    
    # 转换为tensor
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
    
    # 添加batch维度并移到GPU
    img1_tensor = img1_tensor.unsqueeze(0).to(lpips_model.parameters().__next__().device)
    img2_tensor = img2_tensor.unsqueeze(0).to(lpips_model.parameters().__next__().device)
    
    # 计算LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(img1_tensor, img2_tensor)
    
    return lpips_value.item()

def main():
    parser = argparse.ArgumentParser(description="评估DIV2K-Val验证集推理结果")
    parser.add_argument("--gt_dir", type=str, required=True, help="GT图像目录")
    parser.add_argument("--sr_dir", type=str, required=True, help="超分辨率结果目录")
    parser.add_argument("--scale_meta_path", type=str, help="scale元数据文件路径")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="评估结果输出文件")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.gt_dir):
        raise FileNotFoundError(f"GT目录不存在: {args.gt_dir}")
    if not os.path.exists(args.sr_dir):
        raise FileNotFoundError(f"SR目录不存在: {args.sr_dir}")
    
    # 获取图像列表
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.png")))
    sr_files = sorted(glob.glob(os.path.join(args.sr_dir, "*.png")))
    
    print(f"找到 {len(gt_files)} 张GT图像")
    print(f"找到 {len(sr_files)} 张SR图像")
    
    if len(gt_files) != len(sr_files):
        print(f"警告: GT图像数量({len(gt_files)})与SR图像数量({len(sr_files)})不一致")
        # 只处理数量较少的那一组
        min_count = min(len(gt_files), len(sr_files))
        gt_files = gt_files[:min_count]
        sr_files = sr_files[:min_count]
        print(f"将处理前 {min_count} 张图像")
    
    # 加载scale元数据
    scale_dict = {}
    if args.scale_meta_path and os.path.exists(args.scale_meta_path):
        with open(args.scale_meta_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                scale_dict[data['id']] = data['scale']
        print(f"加载了 {len(scale_dict)} 个scale元数据")
    
    # 初始化LPIPS模型
    lpips_model = None
    if HAS_LPIPS:
        lpips_model = lpips.LPIPS(net='alex').to(args.device)
        print("LPIPS模型初始化完成")
    
    # 评估指标
    psnr_values = []
    ssim_values = []
    lpips_values = []
    scale_values = []
    
    print("开始评估...")
    for i, (gt_file, sr_file) in enumerate(tqdm(zip(gt_files, sr_files), total=len(gt_files))):
        # 获取图像ID
        gt_name = os.path.splitext(os.path.basename(gt_file))[0]
        sr_name = os.path.splitext(os.path.basename(sr_file))[0]
        
        if gt_name != sr_name:
            print(f"警告: GT图像名({gt_name})与SR图像名({sr_name})不匹配")
        
        # 加载图像
        gt_img = load_image(gt_file)
        sr_img = load_image(sr_file)
        
        # 确保图像尺寸一致
        if gt_img.shape != sr_img.shape:
            print(f"警告: 图像 {gt_name} 尺寸不匹配: GT{gt_img.shape} vs SR{sr_img.shape}")
            # 将SR图像resize到GT尺寸
            sr_img = np.array(Image.fromarray(sr_img).resize((gt_img.shape[1], gt_img.shape[0]), Image.BICUBIC))
        
        # 计算指标
        psnr_val = calculate_psnr(gt_img, sr_img)
        ssim_val = calculate_ssim(gt_img, sr_img)
        lpips_val = calculate_lpips(gt_img, sr_img, lpips_model)
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)
        
        # 记录scale值
        if gt_name in scale_dict:
            scale_values.append(scale_dict[gt_name])
        else:
            scale_values.append(None)
        
        # 每100张图像打印一次进度
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{len(gt_files)} 张图像")
    
    # 计算统计结果
    results = {
        "total_images": len(gt_files),
        "metrics": {}
    }
    
    if HAS_SKIMAGE and psnr_values:
        results["metrics"]["PSNR"] = {
            "mean": float(np.mean(psnr_values)),
            "std": float(np.std(psnr_values)),
            "min": float(np.min(psnr_values)),
            "max": float(np.max(psnr_values))
        }
    
    if HAS_SKIMAGE and ssim_values:
        results["metrics"]["SSIM"] = {
            "mean": float(np.mean(ssim_values)),
            "std": float(np.std(ssim_values)),
            "min": float(np.min(ssim_values)),
            "max": float(np.max(ssim_values))
        }
    
    if HAS_LPIPS and lpips_values:
        results["metrics"]["LPIPS"] = {
            "mean": float(np.mean(lpips_values)),
            "std": float(np.std(lpips_values)),
            "min": float(np.min(lpips_values)),
            "max": float(np.max(lpips_values))
        }
    
    # 按scale分组统计
    if scale_values and any(s is not None for s in scale_values):
        scale_groups = {}
        for i, scale in enumerate(scale_values):
            if scale is not None:
                scale_key = f"scale_{scale:.3f}"
                if scale_key not in scale_groups:
                    scale_groups[scale_key] = {"psnr": [], "ssim": [], "lpips": []}
                scale_groups[scale_key]["psnr"].append(psnr_values[i])
                scale_groups[scale_key]["ssim"].append(ssim_values[i])
                scale_groups[scale_key]["lpips"].append(lpips_values[i])
        
        results["scale_groups"] = {}
        for scale_key, values in scale_groups.items():
            results["scale_groups"][scale_key] = {
                "count": len(values["psnr"]),
                "PSNR_mean": float(np.mean(values["psnr"])) if values["psnr"] else 0.0,
                "SSIM_mean": float(np.mean(values["ssim"])) if values["ssim"] else 0.0,
                "LPIPS_mean": float(np.mean(values["lpips"])) if values["lpips"] else 0.0
            }
    
    # 保存结果
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果:")
    print("="*50)
    print(f"总图像数量: {results['total_images']}")
    
    for metric_name, metric_data in results["metrics"].items():
        print(f"\n{metric_name}:")
        print(f"  平均值: {metric_data['mean']:.4f}")
        print(f"  标准差: {metric_data['std']:.4f}")
        print(f"  最小值: {metric_data['min']:.4f}")
        print(f"  最大值: {metric_data['max']:.4f}")
    
    if "scale_groups" in results:
        print(f"\n按Scale分组统计:")
        for scale_key, scale_data in results["scale_groups"].items():
            print(f"  {scale_key}: {scale_data['count']} 张图像")
            if scale_data["PSNR_mean"] > 0:
                print(f"    PSNR: {scale_data['PSNR_mean']:.4f}")
            if scale_data["SSIM_mean"] > 0:
                print(f"    SSIM: {scale_data['SSIM_mean']:.4f}")
            if scale_data["LPIPS_mean"] > 0:
                print(f"    LPIPS: {scale_data['LPIPS_mean']:.4f}")
    
    print(f"\n详细结果已保存到: {args.output_file}")

if __name__ == "__main__":
    main()

