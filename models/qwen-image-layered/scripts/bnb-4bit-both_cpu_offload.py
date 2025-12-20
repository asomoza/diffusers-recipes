import torch
from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image
from transformers import Qwen2_5_VLForConditionalGeneration


torch_dtype = torch.bfloat16
device = "cuda"

transformer = QwenImageTransformer2DModel.from_pretrained(
    "OzzyGT/qwen-image-layered-bnb-4bit-transformer", torch_dtype=torch_dtype
)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "OzzyGT/qwen-image-layered-bnb-4bit-text-encoder", torch_dtype=torch_dtype
)

pipe = QwenImageLayeredPipeline.from_pretrained(
    "Qwen/Qwen-Image-Layered", transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload()

image = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-layered/20251220124407_2987430379.png"
).convert("RGBA")

image = pipe(
    image=image,
    negative_prompt=" ",
    generator=torch.Generator("cuda").manual_seed(42),
    cfg_normalize=True,
    use_en_prompt=True,
).images[0]

for i, image in enumerate(image):
    image.save(f"{i}.png")
