#!/bin/bash

# 生成DIV2K-Val验证集脚本
# 按照SeeSR原论文方式：从DIV2K验证集随机裁剪3K个512×512的patches，应用相同的退化pipeline

# 设置参数
DIV2K_VALID_HR="/data_center/data1/dataset/DIV2K/valid/valid_HR"  # DIV2K验证集HR路径
OUTPUT_ROOT="/data4/huangsiyu/SeeSR_baseline/inference_data/DIV2K-val"  # 输出根目录
NUM_PATCHES=3000  # 生成的patch数量（原论文使用3K）
PATCH_SIZE=512    # patch尺寸（原论文使用512×512）
BATCH_SIZE=2      # batch size
NUM_WORKERS=4     # 数据加载器worker数量
SEED=42           # 随机种子

echo "开始生成DIV2K-Val验证集..."
echo "DIV2K验证集HR路径: $DIV2K_VALID_HR"
echo "输出目录: $OUTPUT_ROOT"
echo "生成 $NUM_PATCHES 个 ${PATCH_SIZE}x${PATCH_SIZE} 的patches"
echo "随机种子: $SEED"

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"

# 运行数据生成脚本
CUDA_VISIBLE_DEVICES=0 python /data4/huangsiyu/SeeSR_baseline/utils_data/make_val_dataset.py \
    --hr_dir "$DIV2K_VALID_HR" \
    --out_root "$OUTPUT_ROOT" \
    --mode val \
    --num_patches $NUM_PATCHES \
    --gt_patch $PATCH_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --seed $SEED

echo "DIV2K-Val验证集生成完成！"
echo "输出结构："
echo "  $OUTPUT_ROOT/gt/          # GT patches (512x512)"
echo "  $OUTPUT_ROOT/sr_bicubic/  # LR patches (上采样到512x512)"
echo "  $OUTPUT_ROOT/scale_meta.jsonl  # 元数据文件"
echo ""
echo "下一步："
echo "1. 运行打标脚本生成标签文件"
echo "2. 使用LR图像进行SeeSR推理"
echo "3. 计算推理结果与GT的指标"
