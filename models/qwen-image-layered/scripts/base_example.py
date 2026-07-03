import os

import torch
from diffusers import QwenImageLayeredPipeline
from diffusers.utils import load_image


device = "cuda"
dtype = torch.bfloat16
repo_id = "Qwen/Qwen-Image-Layered"
output_dir = "./outputs/qwen-image-layered"
seed = None

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

pipe = QwenImageLayeredPipeline.from_pretrained(repo_id, torch_dtype=dtype)
pipe.to(device)

image = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-layered/20251220124407_2987430379.png"
).convert("RGBA")

inputs = {
    "image": image,
    "generator": generator,
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
    "num_images_per_prompt": 1,
    "layers": 4,
    "resolution": 640,  # Using different bucket (640, 1024) to determine the resolution. For this version, 640 is recommended
    "cfg_normalize": True,  # Whether enable cfg normalization.
    "use_en_prompt": True,  # Automatic caption language if user does not provide caption
}

image = pipe(**inputs).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, image in enumerate(image):
    image.save(os.path.join(output_dir, f"base_example_{i}_{seed}.png"))
