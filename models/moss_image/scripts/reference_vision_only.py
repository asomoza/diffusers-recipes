import os

import torch
from diffusers import ModularPipelineBlocks
from diffusers.utils import load_image
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model


device = "cuda"
dtype = torch.bfloat16
SDNQ_BITS = 8  # 4 or 8
repo_id = f"OzzyGT/MOSS_Image_SFT_sdnq_dynamic_{SDNQ_BITS}bit"
blocks_repo_id = "OzzyGT/moss_image_custom_blocks"
output_dir = "./outputs/moss_image"
seed = None
prompt = "photo"

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
# the reference workflow also loads the Qwen3.5 vision encoder; the VAE runs in float32, like the original
pipe.load_components(workflow="reference", dtype={"default": dtype, "vae": torch.float32})

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

pipe.to(device)

reference = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/MOSS_Image_Base/bruno-bd-fVWHdvN0K-M-unsplash_cropped.jpg"
)

image = pipe(
    prompt=prompt,
    reference_images=reference,
    reference_max_pixels=1024 * 1024,  # the vision encoder's view of each reference, 1 MP gives 1024 image tokens
    height=1024,
    width=1024,
    num_inference_steps=28,  # the official recommendation is 50 or more
    generator=generator,
    output="images",
)[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"reference_vision_only_{SDNQ_BITS}bit_{seed}.png"))
