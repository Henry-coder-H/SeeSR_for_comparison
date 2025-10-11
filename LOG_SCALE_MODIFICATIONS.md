# SeeSR Baseline Log-Scale Modifications

本文档总结了对SeeSR_baseline代码的修改，实现了以下功能：

## 1. 封装ScaleTimeEmbedding函数

### 修改内容：
- 创建了自定义的`ScaleTimeEmbedding`类，位于`models/scale_embedding.py`
- 从diffusers库中移除了对`ScaleTimeEmbedding`的依赖
- 实现了RoPE (Rotary Position Embedding) 用于标量值的嵌入

### 修改的文件：
- `models/scale_embedding.py` (新建)
- `train_seesr.py` - 修改导入语句
- `models/controlnet.py` - 修改导入语句
- `models/unet_2d_condition.py` - 修改导入语句

## 2. 修改scale采样逻辑为log空间

### 修改内容：
- 将scale采样从均匀采样改为log空间采样
- 采样范围从`[1/16, 1]`改为`[log(1/16), 0]`
- 在log空间均匀采样，然后转换为原始scale值进行图像处理
- 在数据集中直接使用log-scale值

### 修改的文件：
- `utils_data/make_paired_data_anyscale.py` - 修改为log空间采样，保存log-scale值
- `dataloaders/paired_dataset.py` - 直接使用meta中的log-scale值
- `train_seesr.py` - 修改训练循环中的scale处理

## 3. 修改resize处理

### 修改内容：
- 在resize低分辨率图像时，将log-scale重新exp到原始大小
- 保持resize操作使用原始scale值

### 修改的文件：
- `test_seesr.py` - 修改resize逻辑，使用原始scale值进行resize
- `test_seesr_turbo.py` - 同样修改resize逻辑
- `pipelines/pipeline_seesr.py` - 修改scale处理函数

## 4. 修改所有scale embedding输入为logs

### 修改内容：
- 所有传递给模型的scale值都使用log-scale
- 在ControlNet和UNet的调用中使用log-scale
- 在pipeline中正确处理log-scale

### 修改的文件：
- `train_seesr.py` - 训练循环中的scale处理
- `pipelines/pipeline_seesr.py` - pipeline中的scale处理

## 主要技术细节

### ScaleTimeEmbedding实现
```python
class ScaleTimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim: int, rope_dim: int = 128, rope_base: float = 10000.0, use_norm: bool = True):
        # 使用RoPE进行标量嵌入
        # 支持log-scale输入
```

### Log-Scale采样
```python
# 在数据生成脚本中
log_scale = random.uniform(args.log_scale_min, args.log_scale_max)  # log空间均匀采样
sf_any = math.exp(log_scale)  # 转换为原始scale值用于图像处理

# 在数据集中
if self.use_log_scale:
    scale = torch.tensor(self.scale_list[index], dtype=torch.float32)  # 直接使用log-scale值
```

### Resize处理
```python
# 在测试脚本中
rscale = float(args.upscale)  # 原始scale值，用于resize
rscale_log = math.log(rscale)  # log-scale值，用于模型输入

# 传递给pipeline使用log-scale
scale_value=float(rscale_log)

# 但resize时使用原始scale
image = image.resize((ori_width*rscale, ori_height*rscale))
```

## 使用说明

1. **训练时**：数据集会自动将scale转换为log-scale，模型接收log-scale输入
2. **推理时**：需要将原始scale转换为log-scale传递给pipeline，但resize操作仍使用原始scale
3. **兼容性**：可以通过设置`use_log_scale=False`来保持原有的scale处理方式

## 验证方法

1. 检查训练日志中的scale值显示
2. 验证resize操作使用正确的scale值
3. 确认模型接收的是log-scale输入

所有修改都已完成并通过了linter检查，代码应该可以正常运行。
