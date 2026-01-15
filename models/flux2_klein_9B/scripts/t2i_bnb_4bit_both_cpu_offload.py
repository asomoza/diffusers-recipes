import os

import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from transformers import Qwen3ForCausalLM


device = "cuda"
dtype = torch.bfloat16

transformer = Flux2Transformer2DModel.from_pretrained(
    "OzzyGT/flux2_klein_9B_bnb_4bit_transformer", torch_dtype=dtype, device_map="cpu"
)

text_encoder = Qwen3ForCausalLM.from_pretrained(
    "OzzyGT/flux2_klein_9B_bnb_4bit_text_encoder", torch_dtype=dtype, device_map="cpu"
)

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype
)
pipe.enable_model_cpu_offload()

prompt = "photo of a capybara riding a skateboard straight down the middle of a sunny city street, shot from a low-angle, the capybara wears an obviously fake, windblown blonde wig, sunglasses, oversized skater shorts, and scuffed skate shoes, paws balanced confidently on the board; cinematic midday lighting with crisp shadows, shallow depth of field with softly blurred buildings and pedestrians in the background, subtle motion blur on the skateboard wheels; “Diffusers” bold graffiti painted on the asphalt; humorous contrast between the capybara's calm, stoic expression and the high-energy skate culture vibe."
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

if not os.path.exists("./outputs/flux2_klein_9B"):
    os.makedirs("./outputs/flux2_klein_9B")

image.save("./outputs/flux2_klein_9B/t2i_bnb_4bit_both_cpu_offload.png")
