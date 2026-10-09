import os

import torch
from diffusers import QwenImage21Pipeline
from qwen_image_21_vae_strips import patch_vae


device = "cuda"
dtype = torch.bfloat16
repo_id = "Qwen/Qwen-Image-2.1"
output_dir = "./outputs/qwen_image_2_1"
seed = 42
prompt = """The image is a wide realistic photograph of a sign painter lettering a glass shop door at dusk, set against a deep blue-grey cobbled street. The background is a narrow street receding toward the upper right, its wet cobblestones a mottled blue-grey broken by scattered amber highlights, the far end dissolving into soft bokeh. In the centre of the frame a man in his late forties kneels on one knee on the wet stones, his body angled three-quarters away from the viewer and his face in lost profile as he looks up at his work. He wears a heavy charcoal wool coat over a collarless off-white shirt, both flecked with dried paint, and his forearms are bare to the elbow. In his right hand he holds a fine sable lettering brush, its tip loaded with gold size, and his left forearm rests along a slender wooden mahl stick braced against the door frame. Across the upper third of the glass, a row of gold leaf capitals reads "HALVARD & CO.", each letter outlined in a thin dark keyline. Directly beneath it, in smaller black serif capitals, a second line reads "EST. 1931". He is working on the outside face of the glass, so both lines read correctly to the viewer, the wet gold still glossy where the last stroke ended mid-letter. On the right side of the frame, the dark glass returns a reflection of the street behind the viewer: a deep red neon sign whose letters appear reversed as "BAR", and the smeared white headlights of a passing car drawn into horizontal streaks. In the lower-left corner an open wooden toolbox sits on the cobbles with its lid folded back, and beside it three brushes stand upright in a squat glass jar half-filled with what appears to be cloudy solvent. Along the lower edge of the frame, the wet stones throw the gold lettering back as a broken vertical smear, its fragments displaced by the gaps between the cobbles. Behind his left shoulder, a cast-iron downpipe runs up the corner of the building, its paint blistered and rust-streaked. In the upper-left corner, the edge of a striped fabric awning is cropped by the frame. His breath is faintly visible in the cold air, a pale wisp drifting to the right of his jaw. The lighting is low and directional, a warm sodium streetlamp out of frame to the left raking across the glass and the painter's shoulder while the blue dusk fills the shadows, leaving a hard specular line down the edge of the door. The overall composition is asymmetric and horizontal, weighted to the left by the kneeling figure and balanced by the lettering and reflections on the right, in a muted palette of blue-grey, charcoal and warm amber."""

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

pipe = QwenImage21Pipeline.from_pretrained(repo_id, torch_dtype=dtype)
pipe.enable_model_cpu_offload()
# NOTE: workaround for the current diffusers version, whose VAE tiling leaves visible seams.
# Decodes in exact row strips instead, see qwen_image_21_vae_strips.py.
patch_vae(pipe.vae)

image = pipe(
    prompt=prompt,
    height=1696,
    width=2528,
    num_inference_steps=25,
    true_cfg_scale=1.0,
    generator=generator,
).images[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"base_example_vae_fix_{seed}.png"))
