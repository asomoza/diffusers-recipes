import os

import torch
from diffusers import ErnieImagePipeline


device = "cuda"
dtype = torch.bfloat16

MODEL_PATH = "Baidu/ERNIE-Image"
USE_PROMPT_ENHANCER = True

width = 1264
height = 848
seed = None

prompt = """
A highly detailed close-up scene of a middle-aged marine biologist sitting on a weathered dock beside a
large sea lion resting partially out of the water. The man wears a faded blue jacket with a stitched
name tag reading "Dr. Elias Moreno" and rubber boots marked "Harbor Lab Unit 3" along the sides. A
waterproof tablet hangs from a strap across his chest, displaying a screen labeled "Specimen Log v2.4".

The sea lion's whiskers glisten with moisture, and a small yellow tag attached to its fin clearly reads
"SL-207". Around them, ropes, buoys, and fishing nets are loosely arranged across the dock, including a
red buoy labeled "Zone C". A metal bucket filled with fish sits nearby with the printed text "BAIT ONLY",
and a clipboard rests on the wood with handwritten notes titled "Feeding Schedule - Morning Session".

In the background, wooden posts rise from the water toward a foggy harbor where a distant boat shows the
name "North Star" painted along its side. A warning sign nailed to one post reads "CAUTION: WET SURFACE".
The lighting is cool and diffused, with subtle highlights on wet surfaces and muted shadows stretching
along the dock. Fine details such as fabric creases, water droplets, and the texture of the sea lion's
skin are clearly defined.
"""

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
    print(f"Using random seed: {seed}")
generator = torch.Generator(device="cpu").manual_seed(seed)


pipe = ErnieImagePipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_inference_steps=50,
    guidance_scale=4.0,
    use_pe=USE_PROMPT_ENHANCER,
    generator=generator,
).images[0]

os.makedirs("outputs/ernie_image", exist_ok=True)
output_path = f"outputs/ernie_image/base_{width}x{height}_{seed}.png"
image.save(output_path)
