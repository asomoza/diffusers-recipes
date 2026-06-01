import os

import torch
from diffusers import Flux2Pipeline


repo_id = "black-forest-labs/FLUX.2-dev"
device = "cuda"
dtype = torch.bfloat16
output_dir = "./outputs/flux2_dev"
seed = None
prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

pipe = Flux2Pipeline.from_pretrained(repo_id, torch_dtype=dtype)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt=prompt,
    generator=generator,
    num_inference_steps=28,
    guidance_scale=4,
).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"base_t2i_example_{seed}.png"))
