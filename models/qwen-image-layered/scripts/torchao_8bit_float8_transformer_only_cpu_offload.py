import os

import torch
from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image
from transformers import Qwen2_5_VLForConditionalGeneration


device = "cuda"
dtype = torch.bfloat16
repo_id = "Qwen/Qwen-Image-Layered"
output_dir = "./outputs/qwen-image-layered"
seed = None

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

transformer = QwenImageTransformer2DModel.from_pretrained(
    "OzzyGT/qwen-image-layered-torchao-float8-transformer", torch_dtype=dtype, use_safetensors=False
)

text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    repo_id,
    subfolder="text_encoder",
    dtype=dtype,
)

pipe = QwenImageLayeredPipeline.from_pretrained(repo_id, transformer=transformer, torch_dtype=dtype)
pipe.enable_model_cpu_offload()

image = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-layered/20251220124407_2987430379.png"
).convert("RGBA")

image = pipe(
    image=image,
    negative_prompt=" ",
    generator=generator,
    cfg_normalize=True,
    use_en_prompt=True,
).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, image in enumerate(image):
    image.save(os.path.join(output_dir, f"torchao_8bit_float8_transformer_only_cpu_offload_{i}_{seed}.png"))
