import os

import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image


device = "cuda"
dtype = torch.bfloat16
repo_id = "Qwen/Qwen-Image-Edit-2511"
output_dir = "./outputs/qwen_image_edit_2511"
seed = None
prompt = "the turtle from image 1 and the rabbit from image 2 are fighting in an epic battle scene at a beach in a tropical island, 35mm, depth of field, 50mm lens, f/3.5, cinematic lighting"

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

pipe = QwenImageEditPlusPipeline.from_pretrained(repo_id, torch_dtype=dtype)
pipe.to(device)

image1 = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-edit-plus/20251223141129.png"
)
image2 = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-edit-plus/20251223141332.png"
)

image = pipe(
    image=[image1, image2],
    prompt=prompt,
    negative_prompt=" ",
    num_inference_steps=40,
    true_cfg_scale=4.0,
    generator=generator,
).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"base_example_{seed}.png"))
