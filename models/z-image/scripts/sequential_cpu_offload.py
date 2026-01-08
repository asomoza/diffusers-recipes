import os

import torch
from diffusers import ZImagePipeline


pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload()

prompt = "A classroom setting with a large green chalkboard on a wooden frame, illuminated by soft morning light from a window on the left. On the chalkboard, written in clear white chalk handwriting: 'Welcome to Diffusers', 'the library that empowers you to create, customize, and experiment with state of the art diffusion models.' Additional chalk notes appear underneath, illustrating what's possible like 'text-to-image', 'image-to-image', 'inpainting', and 'fine-tuning' sketched alongside tiny doodles of gears, sparkles, and miniature image frames. Wooden desks and chairs fill the room, each desk containing a natural scatter of notebooks, textbooks, and pencils. A couple of open books reveal diagrams of AI model architecture and diffusion processes. The walls are decorated with educational posters—some depicting mathematical formulas. Color scheme: muted greens, warm browns, and crisp white. Shallow depth of field emphasizes the chalkboard."

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

if not os.path.exists("./outputs/zimage"):
    os.makedirs("./outputs/zimage")

image.save("./outputs/zimage/sequential_cpu_offload.png")
