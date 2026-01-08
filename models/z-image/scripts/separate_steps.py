import gc
import os

import torch
from diffusers import AutoencoderKL, ZImagePipeline
from diffusers.image_processor import VaeImageProcessor


# Text encoding step, can use a separate GPU (~11GB VRAM):
embeds_pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    transformer=None,
    vae=None,
).to("cuda")

prompt = "A classroom setting with a large green chalkboard on a wooden frame, illuminated by soft morning light from a window on the left. On the chalkboard, written in clear white chalk handwriting: 'Welcome to Diffusers', 'the library that empowers you to create, customize, and experiment with state of the art diffusion models.' Additional chalk notes appear underneath, illustrating what's possible like 'text-to-image', 'image-to-image', 'inpainting', and 'fine-tuning' sketched alongside tiny doodles of gears, sparkles, and miniature image frames. Wooden desks and chairs fill the room, each desk containing a natural scatter of notebooks, textbooks, and pencils. A couple of open books reveal diagrams of AI model architecture and diffusion processes. The walls are decorated with educational posters—some depicting mathematical formulas. Color scheme: muted greens, warm browns, and crisp white. Shallow depth of field emphasizes the chalkboard."

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
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    text_encoder=None,
    vae=None,
).to("cuda")

prompt_embeds = [embed.to("cuda") for embed in prompt_embeds]

latents = denoise_pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
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
    "Tongyi-MAI/Z-Image-Turbo",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

latents = latents.to(vae.dtype)
latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

with torch.inference_mode():
    image = vae.decode(latents, return_dict=False)[0]

image_processor = VaeImageProcessor(vae_scale_factor=16)
image = image_processor.postprocess(image, output_type="pil")

if not os.path.exists("./outputs/zimage"):
    os.makedirs("./outputs/zimage")

image[0].save("./outputs/zimage/separate_steps.png")
