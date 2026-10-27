import json
import os

import torch
from diffusers import ModularPipelineBlocks


device = "cuda"
dtype = torch.bfloat16
repo_id = "OzzyGT/MOSS_Image_SFT_diffusers"
blocks_repo_id = "OzzyGT/moss_image_custom_blocks"
output_dir = "./outputs/moss_image"
seed = 7

# layout prompt: boxes are [y_min, x_min, y_max, x_max] in pixels with the origin at the center of the canvas
layout = {
    "summary": {
        "meta": ["photo"],
        "canvas": "single",
        "caption": (
            "A photograph of the inside of a small bicycle repair workshop, shot on a Sony A7 IV with a 35mm f/1.8 "
            "lens at f/2.8, 1/125 s, ISO 1600, handheld at eye level, wide 16:9 frame. Low late-afternoon sunlight "
            "comes through a large front window on the right in hard shafts with dust in the air, the rest of the "
            "room is lit by bare fluorescent tubes on the ceiling. Shallow depth of field with the back wall slightly "
            "out of focus, visible sensor grain. A red road bicycle is clamped in a repair stand in the center while "
            "an adult mechanic in a grease-stained navy work apron adjusts its rear derailleur. A painted wooden sign "
            'on the back wall reads "SPOKE & CHAIN". A pegboard of hand tools hangs on the left wall above an A-frame '
            "chalkboard with a price list, and an adult black cat sleeps curled up on a metal stool by the window. "
            "Oil stains on a grey concrete floor, spare tires hanging from hooks, scuffed white walls."
        ),
    },
    "items": [
        {
            "label": "text",
            "caption": "A long painted wooden sign on the back wall, dark green with cream serif capital letters.",
            "text": "SPOKE & CHAIN",
            "bbox": [-496, -464, -346, 236],
        },
        {
            "label": "object",
            "caption": "A perforated pegboard with wrenches, screwdrivers, tire levers and a chain whip on hooks.",
            "bbox": [-396, -984, 124, -544],
        },
        {
            "label": "object",
            "caption": "A red steel road bicycle with drop handlebars, clamped by its seat post in a grey repair stand.",
            "bbox": [-196, -504, 329, 371],
        },
        {
            "label": "object",
            "caption": (
                "An adult man in his forties with short grey hair, a grey t-shirt and a grease-stained navy work "
                "apron, standing and turning a small screwdriver on the rear derailleur."
            ),
            "bbox": [-371, 396, 504, 736],
        },
        {
            "label": "text",
            "caption": "A black A-frame chalkboard on the floor with a handwritten price list in white chalk.",
            "text": "FLAT REPAIR $12\nFULL TUNE-UP $45\nOPEN 8-6",
            "bbox": [184, -964, 524, -604],
        },
        {
            "label": "object",
            "caption": "An adult black cat sleeping curled up on the seat of a metal shop stool in a patch of sunlight.",
            "bbox": [104, 766, 209, 981],
        },
    ],
    "bbox_coord_space": "abs_centralized",
    "bbox_format": "yxyx",
}
prompt = json.dumps(layout, ensure_ascii=False)

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

# trust_remote_code runs the code of the blocks repository and trust_components_code runs the code shipped in
# repo_id, only use them with repositories you trust
blocks = ModularPipelineBlocks.from_pretrained(
    blocks_repo_id,
    trust_remote_code=True,
    components_repo=repo_id,
    trust_components_code=True,
)
pipe = blocks.init_pipeline()
# text-to-image only, which skips the vision encoder of the reference workflow; the VAE runs in float32, like the original
pipe.load_components(workflow="text2image", dtype={"default": dtype, "vae": torch.float32})
pipe.to(device)

image = pipe(
    prompt=prompt,
    height=1152,
    width=2048,
    num_inference_steps=28,  # the official recommendation is 50 or more
    generator=generator,
    output="images",
)[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"base_example_{seed}.png"))
