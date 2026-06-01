import os

import torch
from diffusers import Flux2Pipeline
from sdnq import SDNQConfig  # noqa: F401 — registers SDNQ weight loader
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model


device = "cuda"
dtype = torch.bfloat16
SDNQ_BITS = 4  # 4 or 8
repo_id = f"OzzyGT/FLUX2_dev_sdnq_dynamic_{SDNQ_BITS}bit"
output_dir = "./outputs/flux2_dev"
seed = None
prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

pipe = Flux2Pipeline.from_pretrained(repo_id, torch_dtype=dtype)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

pipe.enable_model_cpu_offload()

image = pipe(
    prompt=prompt,
    generator=generator,
    num_inference_steps=28,
    guidance_scale=4,
).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"t2i_sdnq_{SDNQ_BITS}bit_scoredriven_cpu_offload_{seed}.png"))
