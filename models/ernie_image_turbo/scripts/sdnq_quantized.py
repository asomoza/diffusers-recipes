import os

import torch
from diffusers import ErnieImagePipeline, ErnieImageTransformer2DModel
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from transformers import Ministral3ForCausalLM, Mistral3Model


device = "cuda"
dtype = torch.bfloat16

MODEL_PATH = "Baidu/ERNIE-Image-Turbo"
SDNQ_4BIT_MODEL_PATH = "OzzyGT/ERNIE_Image_Turbo_sdnq_dynamic_int4"
SDNQ_8BIT_MODEL_PATH = "OzzyGT/ERNIE_Image_Turbo_sdnq_dynamic_int8"
SDNQ_BITS = 4  # 4 or 8
SDNQ_MODEL_PATH = SDNQ_4BIT_MODEL_PATH if SDNQ_BITS == 4 else SDNQ_8BIT_MODEL_PATH
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

transformer = ErnieImageTransformer2DModel.from_pretrained(
    SDNQ_MODEL_PATH, subfolder="transformer", torch_dtype=dtype, device_map="cpu"
)

text_encoder = Mistral3Model.from_pretrained(
    SDNQ_MODEL_PATH, subfolder="text_encoder", torch_dtype=dtype, device_map="cpu"
)

if USE_PROMPT_ENHANCER:
    prompt_enhancer = Ministral3ForCausalLM.from_pretrained(
        SDNQ_MODEL_PATH, subfolder="pe", torch_dtype=dtype, device_map="cpu"
    )
else:
    prompt_enhancer = None

pipe = ErnieImagePipeline.from_pretrained(
    MODEL_PATH,
    transformer=transformer,
    text_encoder=text_encoder,
    pe=prompt_enhancer,
    torch_dtype=torch.bfloat16,
)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
    if USE_PROMPT_ENHANCER:
        pipe.prompt_enhancer = apply_sdnq_options_to_model(pipe.pe, use_quantized_matmul=True)

pipe.enable_model_cpu_offload()

image = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_inference_steps=8,
    guidance_scale=1.0,
    use_pe=USE_PROMPT_ENHANCER,
    generator=generator,
).images[0]

os.makedirs("outputs/ernie_image_turbo", exist_ok=True)
output_path = f"outputs/ernie_image_turbo/ernie_sdnq_{SDNQ_BITS}bit_{width}x{height}_{seed}.png"
image.save(output_path)
