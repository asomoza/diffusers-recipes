import os

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks
from diffusers.utils import load_image
from sdnq import SDNQConfig  # noqa: F401


device = "cuda"
dtype = torch.bfloat16
blocks_id = "OzzyGT/krea2_reference_blocks"
repo_id = "OzzyGT/Krea_2_Turbo_sdnq_dynamic_8bit"
lora_id = "conradlocke/krea2-identity-edit"
lora_weight_name = "krea2_identity_edit_v1_2.safetensors"
output_dir = "./outputs/krea2_turbo"
seed = None
prompt = "add sunglasses and a hat to the capybara"

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

blocks = ModularPipelineBlocks.from_pretrained(blocks_id, trust_remote_code=True)
pipe = blocks.init_pipeline(repo_id)
pipe.load_components(
    names=["text_encoder", "tokenizer", "transformer", "vae", "scheduler"],
    dtype=dtype,
)
pipe.load_lora_weights(lora_id, weight_name=lora_weight_name)
pipe.to(device)

reference = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/krea2_reference/alvin-david--cHj_FC6ukw-unsplash_croped.jpg"
)
# mask = load_image(
#     "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/krea2_reference/20260908133430_reference_mask_0_bw.png"
# )

image = pipe(
    prompt=prompt,
    reference_images=reference,
    # reference_masks=mask,
    reference_mode="prepend",
    reference_mask_mode="exclude_blank",
    mask_reference_latents=True,
    grounding_px=768,
    height=1024,
    width=1024,
    num_inference_steps=8,
    generator=generator,
    output="images",
)[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"reference_identity_edit_{seed}.png"))
