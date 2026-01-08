import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image


pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16, device_map="cuda"
)

image1 = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-edit-plus/20251223141129.png"
)
image2 = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-edit-plus/20251223141332.png"
)
image3 = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-edit-plus/20251223141636.png"
)

prompt = "put the turtle, the rabbit and the capybara together in a game show setting."

output = pipe(
    image=[image1, image2, image3],
    prompt=prompt,
    negative_prompt=" ",
    num_inference_steps=40,
    guidance_scale=1.0,
    generator=torch.Generator("cuda").manual_seed(42),
)
output_image = output.images[0]
output_image.save("output_image_edit_2511.png")
