#!/bin/bash

# 为DIV2K-Val验证集生成标签文件
# 使用RAM模型为GT图像生成描述性标签

# 设置参数
VAL_ROOT="/data4/huangsiyu/SeeSR_baseline/inference_data/DIV2K-val"  # DIV2K-Val验证集根目录
RAM_MODEL_PATH="preset/models/ram_swin_large_14m.pth"  # RAM模型路径

# 多GPU处理参数（可选）
START_GPU=0  # 起始GPU索引
ALL_GPU=1    # 总GPU数量

echo "开始为DIV2K-Val验证集生成标签..."
echo "验证集根目录: $VAL_ROOT"
echo "RAM模型路径: $RAM_MODEL_PATH"
echo "GPU配置: GPU $START_GPU / $ALL_GPU"

# 检查路径是否存在
if [ ! -d "$VAL_ROOT" ]; then
    echo "错误: 验证集根目录不存在: $VAL_ROOT"
    echo "请先运行 generate_div2k_val.sh 生成验证集"
    exit 1
fi

if [ ! -d "$VAL_ROOT/gt" ]; then
    echo "错误: GT目录不存在: $VAL_ROOT/gt"
    echo "请先运行 generate_div2k_val.sh 生成验证集"
    exit 1
fi

if [ ! -f "$RAM_MODEL_PATH" ]; then
    echo "错误: RAM模型文件不存在: $RAM_MODEL_PATH"
    exit 1
fi

# 统计GT图像数量
GT_COUNT=$(ls -1 "$VAL_ROOT/gt"/*.png 2>/dev/null | wc -l)
echo "找到 $GT_COUNT 张GT图像"

if [ "$GT_COUNT" -eq 0 ]; then
    echo "错误: 未找到GT图像"
    exit 1
fi

# 运行标签生成脚本
CUDA_VISIBLE_DEVICES=0 python /data4/huangsiyu/SeeSR_baseline/utils_data/make_val_tags.py \
    --val_root "$VAL_ROOT" \
    --start_gpu $START_GPU \
    --all_gpu $ALL_GPU \
    --ram_model_path "$RAM_MODEL_PATH"

echo "标签生成完成！"
echo "标签文件保存在: $VAL_ROOT/tag/"
echo ""
echo "下一步："
echo "1. 检查标签文件是否正确生成"
echo "2. 运行推理脚本进行SeeSR推理"
echo "3. 评估推理结果"


