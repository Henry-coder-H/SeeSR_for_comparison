#!/bin/bash

# DIV2K-Val验证集评估脚本
# 计算SeeSR推理结果与GT图像的PSNR、SSIM、LPIPS等指标

# 设置路径
DIV2K_VAL_ROOT="/data4/huangsiyu/SeeSR_baseline/inference_data/DIV2K-val"
GT_DIR="$DIV2K_VAL_ROOT/gt"                    # GT图像目录
SR_DIR="/data4/huangsiyu/SeeSR_baseline/inference_data/output_HR/sample00"  # 超分辨率结果目录
SCALE_META_PATH="$DIV2K_VAL_ROOT/scale_meta.jsonl"  # scale元数据文件
OUTPUT_FILE="div2k_val_evaluation_results.json"  # 评估结果输出文件

echo "开始评估DIV2K-Val验证集推理结果..."
echo "GT图像目录: $GT_DIR"
echo "SR结果目录: $SR_DIR"
echo "Scale元数据: $SCALE_META_PATH"
echo "输出文件: $OUTPUT_FILE"

# 检查路径是否存在
if [ ! -d "$GT_DIR" ]; then
    echo "错误: GT图像目录不存在: $GT_DIR"
    echo "请先运行 generate_div2k_val.sh 生成验证集"
    exit 1
fi

if [ ! -d "$SR_DIR" ]; then
    echo "错误: SR结果目录不存在: $SR_DIR"
    echo "请先运行 inference_div2k_val.sh 进行推理"
    exit 1
fi

if [ ! -f "$SCALE_META_PATH" ]; then
    echo "警告: Scale元数据文件不存在: $SCALE_META_PATH"
    echo "将跳过按scale分组的统计"
    SCALE_META_PATH=""
fi

# 统计图像数量
GT_COUNT=$(ls -1 "$GT_DIR"/*.png 2>/dev/null | wc -l)
SR_COUNT=$(ls -1 "$SR_DIR"/*.png 2>/dev/null | wc -l)
echo "GT图像数量: $GT_COUNT"
echo "SR图像数量: $SR_COUNT"

if [ "$GT_COUNT" -eq 0 ]; then
    echo "错误: 未找到GT图像"
    exit 1
fi

if [ "$SR_COUNT" -eq 0 ]; then
    echo "错误: 未找到SR图像"
    exit 1
fi

# 运行评估脚本
CUDA_VISIBLE_DEVICES=6 python /data4/huangsiyu/SeeSR_baseline/evaluate_div2k_val.py \
    --gt_dir "$GT_DIR" \
    --sr_dir "$SR_DIR" \
    --scale_meta_path "$SCALE_META_PATH" \
    --output_file "$OUTPUT_FILE" \
    --device cuda

echo "评估完成！"
echo "评估结果已保存到: $OUTPUT_FILE"
echo ""
echo "你可以查看评估结果文件了解详细的指标统计信息。"
