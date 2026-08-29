import os

import torch
from diffusers import ZImagePipeline


device = "cuda"
dtype = torch.bfloat16
repo_id = "Tongyi-MAI/Z-Image-Turbo"
output_dir = "./outputs/zimage"
seed = None
prompt = """Pixel art style, A classroom setting with a large green chalkboard on a wooden frame, illuminated by soft morning light from a window on the left. On the chalkboard, written in clear white chalk handwriting: 'Welcome to Diffusers', 'the library that empowers you to create, customize, and experiment with state of the art diffusion models.' Additional chalk notes appear underneath, illustrating what's possible like 'text-to-image', 'image-to-image', 'inpainting', and 'fine-tuning' sketched alongside tiny doodles of gears, sparkles, and miniature image frames. Wooden desks and chairs fill the room, each desk containing a natural scatter of notebooks, textbooks, and pencils. A couple of open books reveal diagrams of AI model architecture and diffusion processes. The walls are decorated with educational posters—some depicting mathematical formulas. Color scheme: muted greens, warm browns, and crisp white. Shallow depth of field emphasizes the chalkboard."""

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

pipe = ZImagePipeline.from_pretrained(
    repo_id,
    torch_dtype=dtype,
)
pipe.to(device)

pipe.load_lora_weights(
    "tarn59/pixel_art_style_lora_z_image_turbo",
    weight_name="pixel_art_style_z_image_turbo.safetensors",  # if the repo only has one lora weight file, this argument can be omitted
    adapter_name="pixel_art",  # if you're not planning to use set_adapters later, this argument can be omitted
)

# you can repeat this step to load multiple lora weights into different adapters
# pipe.load_lora_weights(
#     "some_repo/lora_name.safetensors",
#     adapter_name="other_lora",
# )

pipe.set_adapters("pixel_art", 1.0)  # set the adapter to use with a specific weight if needed
# pipe.set_adapters(["pixel_art", "other_lora"], [0.7, 0.5])  # example on how to set multiple adapters with different weights

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=generator,
).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"lora_{seed}.png"))
