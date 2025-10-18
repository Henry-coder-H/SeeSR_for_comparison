#!/bin/bash

# SeeSR DIV2K-Val验证集推理脚本
# 使用生成的LR图像进行推理，得到超分辨率结果

# 设置基本参数
SEESR_MODEL_PATH="/data4/huangsiyu/SeeSR_baseline/experience/seesr_logscale/checkpoint-100000"  # 请替换为实际的模型路径
RAM_FT_PATH="/data4/huangsiyu/SeeSR_baseline/preset/models/ram_swin_large_14m.pth"      # 请替换为实际的RAM微调模型路径
PRETRAINED_MODEL_PATH="/data4/huangsiyu/SeeSR_baseline/preset/models/stable-diffusion-2-base"  # 请替换为预训练模型路径

# DIV2K-Val数据集路径
DIV2K_VAL_ROOT="/data4/huangsiyu/SeeSR_baseline/inference_data/DIV2K-val"
LR_IMAGE_PATH="$DIV2K_VAL_ROOT/sr_bicubic"  # LR图像路径
GT_IMAGE_PATH="$DIV2K_VAL_ROOT/gt"          # GT图像路径
TAG_PATH="$DIV2K_VAL_ROOT/tag"              # 标签文件路径
SCALE_META_PATH="$DIV2K_VAL_ROOT/scale_meta.jsonl"  # scale元数据路径

# 输出目录
OUTPUT_DIR="/data4/huangsiyu/SeeSR_baseline/inference_data/output_HR"

# 推理参数
SAMPLE_TIMES=1                # 每个图像生成的样本数
SEED=42                       # 随机种子，确保结果可复现

# 模型推理参数
GUIDANCE_SCALE=5.5
CONDITIONING_SCALE=1.0
NUM_INFERENCE_STEPS=50
PROCESS_SIZE=512

# 内存优化参数（根据GPU显存调整）
VAE_ENCODER_TILED_SIZE=1024
VAE_DECODER_TILED_SIZE=224
LATENT_TILED_SIZE=96
LATENT_TILED_OVERLAP=32

echo "开始SeeSR DIV2K-Val验证集推理..."
echo "LR图像路径: $LR_IMAGE_PATH"
echo "GT图像路径: $GT_IMAGE_PATH"
echo "标签文件路径: $TAG_PATH"
echo "Scale元数据路径: $SCALE_META_PATH"
echo "输出目录: $OUTPUT_DIR"

# 检查路径是否存在
if [ ! -d "$LR_IMAGE_PATH" ]; then
    echo "错误: LR图像路径不存在: $LR_IMAGE_PATH"
    echo "请先运行 generate_div2k_val.sh 生成验证集"
    exit 1
fi

if [ ! -d "$GT_IMAGE_PATH" ]; then
    echo "错误: GT图像路径不存在: $GT_IMAGE_PATH"
    echo "请先运行 generate_div2k_val.sh 生成验证集"
    exit 1
fi

if [ ! -f "$SCALE_META_PATH" ]; then
    echo "错误: Scale元数据文件不存在: $SCALE_META_PATH"
    echo "请先运行 generate_div2k_val.sh 生成验证集"
    exit 1
fi

if [ ! -d "$TAG_PATH" ]; then
    echo "错误: 标签文件目录不存在: $TAG_PATH"
    echo "请先运行 generate_val_tags.sh 生成标签文件"
    exit 1
fi

# 统计图像数量
LR_COUNT=$(ls -1 "$LR_IMAGE_PATH"/*.png 2>/dev/null | wc -l)
GT_COUNT=$(ls -1 "$GT_IMAGE_PATH"/*.png 2>/dev/null | wc -l)
echo "LR图像数量: $LR_COUNT"
echo "GT图像数量: $GT_COUNT"

if [ "$LR_COUNT" -eq 0 ]; then
    echo "错误: 未找到LR图像"
    exit 1
fi

if [ "$LR_COUNT" -ne "$GT_COUNT" ]; then
    echo "警告: LR图像数量($LR_COUNT)与GT图像数量($GT_COUNT)不一致"
fi

CUDA_VISIBLE_DEVICES="4,5" python /data4/huangsiyu/SeeSR_baseline/test_seesr.py \
    --seesr_model_path "$SEESR_MODEL_PATH" \
    --ram_ft_path "$RAM_FT_PATH" \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    --lr_image_path "$LR_IMAGE_PATH" \
    --gt_image_path "$GT_IMAGE_PATH" \
    --tag_path "$TAG_PATH" \
    --scale_meta_path "$SCALE_META_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sample_times $SAMPLE_TIMES \
    --seed $SEED \
    --guidance_scale $GUIDANCE_SCALE \
    --conditioning_scale $CONDITIONING_SCALE \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --process_size $PROCESS_SIZE \
    --vae_encoder_tiled_size $VAE_ENCODER_TILED_SIZE \
    --vae_decoder_tiled_size $VAE_DECODER_TILED_SIZE \
    --latent_tiled_size $LATENT_TILED_SIZE \
    --latent_tiled_overlap $LATENT_TILED_OVERLAP \
    --mixed_precision fp16 \
    --align_method adain \
    --start_point lr \
    --prompt "" \
    --added_prompt "clean, high-resolution, 8k" \
    --negative_prompt "dotted, noise, blur, lowres, smooth" \
    --save_prompts

echo "推理完成！"
echo "推理结果保存在: $OUTPUT_DIR/sample00/"
echo "下一步：运行评估脚本计算指标"
