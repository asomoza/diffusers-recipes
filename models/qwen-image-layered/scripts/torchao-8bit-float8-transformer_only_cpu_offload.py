import os

import torch
from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image
from transformers import Qwen2_5_VLForConditionalGeneration


transformer = QwenImageTransformer2DModel.from_pretrained(
    "OzzyGT/qwen-image-layered-torchao-float8-transformer", torch_dtype=torch.bfloat16, use_safetensors=False
)

text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen-Image-Layered",
    subfolder="text_encoder",
    dtype=torch.bfloat16,
)

pipe = QwenImageLayeredPipeline.from_pretrained(
    "Qwen/Qwen-Image-Layered", transformer=transformer, torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

image = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-layered/20251220124407_2987430379.png"
).convert("RGBA")

image = pipe(
    image=image,
    negative_prompt=" ",
    generator=torch.Generator("cuda").manual_seed(42),
    cfg_normalize=True,
    use_en_prompt=True,
).images[0]

if not os.path.exists("./outputs/qwen-image-layered"):
    os.makedirs("./outputs/qwen-image-layered")

for i, image in enumerate(image):
    image.save(f"./outputs/qwen-image-layered/torchao-8bit-float8-transformer_only_cpu_offload_{i}.png")
