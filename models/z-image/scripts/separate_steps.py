import gc
import os

import torch
from diffusers import AutoencoderKL, ZImagePipeline
from diffusers.image_processor import VaeImageProcessor


device = "cuda"
dtype = torch.bfloat16
repo_id = "Tongyi-MAI/Z-Image-Turbo"
output_dir = "./outputs/zimage"
seed = None
prompt = """A classroom setting with a large green chalkboard on a wooden frame, illuminated by soft morning light from a window on the left. On the chalkboard, written in clear white chalk handwriting: 'Welcome to Diffusers', 'the library that empowers you to create, customize, and experiment with state of the art diffusion models.' Additional chalk notes appear underneath, illustrating what's possible like 'text-to-image', 'image-to-image', 'inpainting', and 'fine-tuning' sketched alongside tiny doodles of gears, sparkles, and miniature image frames. Wooden desks and chairs fill the room, each desk containing a natural scatter of notebooks, textbooks, and pencils. A couple of open books reveal diagrams of AI model architecture and diffusion processes. The walls are decorated with educational posters—some depicting mathematical formulas. Color scheme: muted greens, warm browns, and crisp white. Shallow depth of field emphasizes the chalkboard."""

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

# Text encoding step, can use a separate GPU (~11GB VRAM):
embeds_pipe = ZImagePipeline.from_pretrained(
    repo_id,
    torch_dtype=dtype,
    transformer=None,
    vae=None,
).to(device)

prompt_embeds, negative_prompt_embeds = embeds_pipe.encode_prompt(
    prompt=prompt,
    do_classifier_free_guidance=False,
)

prompt_embeds = [embed.clone().detach() for embed in prompt_embeds]

embeds_pipe.to("cpu")
del embeds_pipe
gc.collect()
torch.cuda.empty_cache()

# Denoising step, requires the most VRAM (~12GB VRAM):
denoise_pipe = ZImagePipeline.from_pretrained(
    repo_id,
    torch_dtype=dtype,
    text_encoder=None,
    vae=None,
).to(device)

prompt_embeds = [embed.to(device) for embed in prompt_embeds]

latents = denoise_pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

latents = latents[0].clone().detach()

denoise_pipe.to("cpu")
del denoise_pipe
gc.collect()
torch.cuda.empty_cache()

# Decoding step, can use a separate GPU (~4GB VRAM):
vae = AutoencoderKL.from_pretrained(
    repo_id,
    subfolder="vae",
    torch_dtype=dtype,
)
vae.to(device)

latents = latents.to(vae.dtype)
latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

with torch.inference_mode():
    image = vae.decode(latents, return_dict=False)[0]

image_processor = VaeImageProcessor(vae_scale_factor=16)
image = image_processor.postprocess(image, output_type="pil")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image[0].save(os.path.join(output_dir, f"separate_steps_{seed}.png"))
