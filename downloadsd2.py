import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base",
    torch_dtype=torch.float16,
    use_safetensors=True,
    resume_download=True
)

pipe.save_pretrained("/data4/huangsiyu/SeeSR/preset/models/stable-diffusion-2-base")
