import os

import torch
from diffusers import QwenImageLayeredPipeline
from diffusers.utils import load_image


pipe = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered", torch_dtype=torch.bfloat16)
pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
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
    image.save(f"./outputs/qwen-image-layered/layerwise_{i}.png")
