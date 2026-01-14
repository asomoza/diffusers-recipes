import os

import torch
from diffusers import GlmImagePipeline, GlmImageTransformer2DModel
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from transformers import GlmImageForConditionalGeneration, T5EncoderModel


torch_dtype = torch.bfloat16


text_encoder = T5EncoderModel.from_pretrained(
    "Disty0/GLM-Image-SDNQ-4bit-dynamic",
    subfolder="text_encoder",
    dtype=torch_dtype,
    device_map="cpu",
)

vision_language_encoder = GlmImageForConditionalGeneration.from_pretrained(
    "Disty0/GLM-Image-SDNQ-4bit-dynamic",
    subfolder="vision_language_encoder",
    torch_dtype=torch_dtype,
    device_map="cpu",
)

transformer = GlmImageTransformer2DModel.from_pretrained(
    "Disty0/GLM-Image-SDNQ-4bit-dynamic",
    subfolder="transformer",
    torch_dtype=torch_dtype,
    device_map="cpu",
)

pipe = GlmImagePipeline.from_pretrained(
    "zai-org/GLM-Image",
    transformer=transformer,
    vision_language_encoder=vision_language_encoder,
    text_encoder=text_encoder,
    torch_dtype=torch.bfloat16,
)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

pipe.enable_model_cpu_offload()

prompt = """Ultra-realistic café storefront scene, cinematic composition, street-level camera angle centered on a glass-front café entrance with a warm wooden frame, early afternoon natural light spilling through the windows, soft reflections on the glass, cozy interior glow, realistic depth of field. On the left side of the entrance, behind the counter and clearly visible from outside, a large vertical green chalkboard menu dominates the scene, slightly dusty chalk texture, hand-drawn white and pastel chalk lettering with subtle imperfections, artisanal style. At the top center of the chalkboard, bold hand-lettered text reads "TODAY'S MENU". Clean, symmetrical menu layout. Left column lists classic coffee drinks with readable text and prices in US dollars: "Espresso — $3.00", "Americano — $3.50", "Cappuccino — $4.50", "Latte — $4.75", "Flat White — $4.50". Right column lists specialty drinks with readable text: "Matcha Latte — $5.25", "Chai Latte — $4.95", "Hot Chocolate — $4.25", "Iced Coffee — $3.95". Bottom section of the chalkboard labeled "PASTRIES" in larger chalk lettering, underlined with a hand-drawn line, followed by pastry items with readable prices: "Croissant — $3.75", "Chocolate Muffin — $3.50", "Banana Bread — $4.00", "Blueberry Scone — $3.75". Decorative chalk illustrations on the board include a steaming coffee cup with heart-shaped steam in the top right corner, a detailed croissant sketch near the pastries section, a small muffin drawing beside "Chocolate Muffin", and delicate leafy vines framing the corners, all in soft pastel chalk tones. Center of the frame shows the café interior through the glass window, two rows of small wooden tables arranged parallel to the storefront. At the front table, a couple sits facing each other, smiling and talking over ceramic cups and small plates. Further inside, a person works on a laptop next to a coffee mug, another group chats quietly at a back table. Right side of the scene includes additional seating near the window, a medium-sized potted plant with lush green leaves, warm pendant lights hanging from the ceiling. Fine details throughout: mismatched wooden and metal chairs, small brass bell above the door, subtle chalk dust on the menu, steam rising from cups, realistic reflections of the street on the glass, calm, inviting, lived-in café atmosphere, high realism, sharp focus, natural color grading."""
image = pipe(
    prompt=prompt,
    height=32 * 32,
    width=36 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

if not os.path.exists("./outputs/glm-image"):
    os.makedirs("./outputs/glm-image")

image.save("./outputs/glm-image/sdnq-4bit-dynamic.png")
