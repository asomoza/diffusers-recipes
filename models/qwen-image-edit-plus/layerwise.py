import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image


pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

image1 = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-edit-plus/20251223141129.png"
)
image2 = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-edit-plus/20251223141332.png"
)
image3 = load_image(
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/qwen-image-edit-plus/20251223141636.png"
)

prompt = "grab the turtle from image 1, the rabbit from image 2 and the capybara from image 3 and put them together in a game show setting while preservening all their features and clothes intact"

output = pipe(
    image=[image1, image2, image3],
    prompt=prompt,
    negative_prompt=" ",
    true_cfg_scale=4.0,
    num_inference_steps=40,
    generator=torch.Generator("cuda").manual_seed(42),
)
output_image = output.images[0]
output_image.save("output_image_edit_2511.png")
