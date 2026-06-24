import os

import torch
from diffusers import Krea2Pipeline


DEVICE = "cuda:0"
dtype = torch.bfloat16
repo_id = "krea/Krea-2-Turbo"
output_dir = "./outputs/krea2_turbo"
seed = None
prompt = """A photorealistic red fox stands atop the rusted roof of an abandoned forest ranger station during a summer thunderstorm.
The fox occupies the upper third of the composition, its rain-soaked orange fur revealing individual strands, darker wet patches,
and subtle variations in texture around the neck and tail. Its amber eyes reflect the distant lightning illuminating the scene.
The weathered metal roof beneath it is covered in peeling paint, oxidized rust patterns, shallow puddles, scattered pine needles,
and riveted seams that lead the eye toward the background. Behind the fox, a dense evergreen forest fades into layers of atmospheric
mist, with tall spruce trees partially obscured by sheets of rainfall. A wooden warning sign mounted near the roof edge displays bold
white lettering reading "FIRE LOOKOUT STATION" above smaller weathered maintenance notices. Cool blue-grey storm clouds dominate the
sky while a warm orange glow from a distant sunset breaks through near the horizon. Cinematic natural lighting creates complex
reflections across the wet surfaces, combining soft overcast shadows with sharp highlights from intermittent lightning. Documentary
wildlife photography aesthetic, shallow telephoto compression, exceptional environmental detail, realistic weather effects, and
high-fidelity material rendering."""

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

pipe = Krea2Pipeline.from_pretrained(repo_id, torch_dtype=torch.bfloat16)
pipe.to(DEVICE)

image = pipe(prompt=prompt, num_inference_steps=8, guidance_scale=1.0, generator=generator).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"base_t2i_example_{seed}.png"))
