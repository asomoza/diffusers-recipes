import os

import torch
from diffusers import QwenImageLayeredPipeline
from PIL import Image


pipeline = QwenImageLayeredPipeline.from_pretrained(
    "Qwen/Qwen-Image-Layered", torch_dtype=torch.bfloat16, device_map="cuda"
)

image = Image.open(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/blob/main/qwen-image-layered/20251220124407_2987430379.png"
).convert("RGBA")

inputs = {
    "image": image,
    "generator": torch.Generator(device="cuda").manual_seed(777),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
    "num_images_per_prompt": 1,
    "layers": 4,
    "resolution": 640,  # Using different bucket (640, 1024) to determine the resolution. For this version, 640 is recommended
    "cfg_normalize": True,  # Whether enable cfg normalization.
    "use_en_prompt": True,  # Automatic caption language if user does not provide caption
}

image = pipeline(**inputs).images[0]

if not os.path.exists("./outputs/qwen-image-layered"):
    os.makedirs("./outputs/qwen-image-layered")

for i, image in enumerate(image):
    image.save(f"./outputs/qwen-image-layered/base_example_{i}.png")
