import os

import torch
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from transformers import Qwen2_5_VLForConditionalGeneration


device = "cuda"
dtype = torch.bfloat16
repo_id = "Qwen/Qwen-Image-Edit-2511"
output_dir = "./outputs/qwen_image_edit_2511"
seed = None
prompt = "the turtle from image 1 and the rabbit from image 2 are fighting in an epic battle scene at a beach in a tropical island, 35mm, depth of field, 50mm lens, f/3.5, cinematic lighting"

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32",
    subfolder="text_encoder",
    dtype=dtype,
    device_map="cpu",
)

transformer = QwenImageTransformer2DModel.from_pretrained(
    "Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32",
    subfolder="transformer",
    torch_dtype=dtype,
    device_map="cpu",
)


pipe = QwenImageEditPlusPipeline.from_pretrained(
    repo_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype
)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

pipe.enable_model_cpu_offload()

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

image.save(os.path.join(output_dir, f"sdnq_uint4_{seed}.png"))
