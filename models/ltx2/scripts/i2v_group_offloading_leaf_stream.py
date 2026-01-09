import os

import torch
from diffusers import LTX2ImageToVideoPipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.utils import load_image


onload_device = torch.device("cuda")
offload_device = torch.device("cpu")


pipe = LTX2ImageToVideoPipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
pipe.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True,
    record_stream=True,
)
pipe.vae.enable_tiling()

image = load_image("https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/license.png")

prompt = "Cinematic video with professional lighting, shot with a 35mm lens and shallow depth of field. The scene opens on an extreme close-up of a real car license plate reading “DIFFUSERS,” captured with macro focus, rock music kicks in and builds intensity. The setting is an outdoor wet mountain road, surrounded by dark silhouettes of hills, trees, and winding asphalt. The car is a high-performance modern sports car with large performance tires gripping the rain-soaked road. A deep combustion engine growl builds as the car suddenly accelerates forward."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

frame_rate = 24.0
video, audio = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=40,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
    output_type="np",
    return_dict=False,
)
video = (video * 255).round().astype("uint8")
video = torch.from_numpy(video)

if not os.path.exists("./outputs/ltx2"):
    os.makedirs("./outputs/ltx2")

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
    output_path="./outputs/ltx2/i2v_group_offloading_leaf_stream.mp4",
)
