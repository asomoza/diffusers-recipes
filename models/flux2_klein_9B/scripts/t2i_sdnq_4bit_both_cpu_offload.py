import os

import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from transformers import Qwen3ForCausalLM


device = "cuda"
dtype = torch.bfloat16
repo_id = "black-forest-labs/FLUX.2-klein-9B"
output_dir = "./outputs/flux2_klein_9B"
seed = None
prompt = "photo of a capybara riding a skateboard straight down the middle of a sunny city street, shot from a low-angle, the capybara wears an obviously fake, windblown blonde wig, sunglasses, oversized skater shorts, and scuffed skate shoes, paws balanced confidently on the board; cinematic midday lighting with crisp shadows, shallow depth of field with softly blurred buildings and pedestrians in the background, subtle motion blur on the skateboard wheels; “Diffusers” bold graffiti painted on the asphalt; humorous contrast between the capybara's calm, stoic expression and the high-energy skate culture vibe."

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

transformer = Flux2Transformer2DModel.from_pretrained(
    "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32", subfolder="transformer", torch_dtype=dtype, device_map="cpu"
)

text_encoder = Qwen3ForCausalLM.from_pretrained(
    "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32", subfolder="text_encoder", torch_dtype=dtype, device_map="cpu"
)

pipe = Flux2KleinPipeline.from_pretrained(
    repo_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype
)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

pipe.enable_model_cpu_offload()

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=generator,
).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"t2i_sdnq_4bit_both_cpu_offload_{seed}.png"))
