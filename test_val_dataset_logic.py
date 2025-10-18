#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试验证数据集生成逻辑
验证scale值是否正确设置为固定值
"""

import json
import os

def test_scale_meta(meta_path):
    """测试scale元数据文件"""
    if not os.path.exists(meta_path):
        print(f"错误: 元数据文件不存在: {meta_path}")
        return False
    
    scales = []
    with open(meta_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            scales.append(data['scale'])
    
    print(f"总共 {len(scales)} 个patches")
    print(f"Scale值范围: {min(scales):.6f} - {max(scales):.6f}")
    print(f"所有scale值是否相同: {len(set(scales)) == 1}")
    
    if len(set(scales)) == 1:
        print(f"固定scale值: {scales[0]:.6f}")
        print(f"对应的原始scale值: {2.718281828459045 ** scales[0]:.6f}")
        return True
    else:
        print("错误: scale值不统一！")
        return False

def main():
    # 测试路径
    meta_path = "/data4/huangsiyu/SeeSR_baseline/inference_data/DIV2K-val/scale_meta.jsonl"
    
    print("测试DIV2K-Val验证集scale元数据...")
    success = test_scale_meta(meta_path)
    
    if success:
        print("✓ 验证集生成逻辑正确：所有patches使用相同的固定scale值")
    else:
        print("✗ 验证集生成逻辑有问题：scale值不统一")

if __name__ == "__main__":
    main()


