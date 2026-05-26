import os

import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.utils import load_image
from sdnq import SDNQConfig
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from transformers import Qwen3ForCausalLM


device = "cuda"
dtype = torch.bfloat16

prompt = "put the subjects from image 1 and image 2 at the beach in an epic fight"
image1 = load_image("https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/resources/turtle.png")
image2 = load_image("https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/resources/kangaroo.png")

transformer = Flux2Transformer2DModel.from_pretrained(
    "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32", subfolder="transformer", torch_dtype=dtype, device_map="cpu"
)

text_encoder = Qwen3ForCausalLM.from_pretrained(
    "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32", subfolder="text_encoder", torch_dtype=dtype, device_map="cpu"
)

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype
)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

pipe.enable_model_cpu_offload()

image = pipe(
    image=[image1, image2],
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

if not os.path.exists("./outputs/flux2_klein_9B"):
    os.makedirs("./outputs/flux2_klein_9B")

image.save("./outputs/flux2_klein_9B/i2i_multiple_sdnq_4bit_both_cpu_offload.png")
