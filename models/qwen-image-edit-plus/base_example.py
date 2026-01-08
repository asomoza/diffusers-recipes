import os

import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image


pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16, device_map="cuda"
)

image1 = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-edit-plus/20251223141129.png"
)
image2 = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-edit-plus/20251223141332.png"
)

prompt = "the turtle from image 1 and the rabbit from image 2 are fighting in an epic battle scene at a beach in a tropical island, 35mm, depth of field, 50mm lens, f/3.5, cinematic lighting"

image = pipe(
    image=[image1, image2],
    prompt=prompt,
    negative_prompt=" ",
    num_inference_steps=40,
    true_cfg_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

if not os.path.exists("./outputs/qwen-image-edit-plus"):
    os.makedirs("./outputs/qwen-image-edit-plus")

image.save("./outputs/qwen-image-edit-plus/layerwise.png")
