import os

import torch
from diffusers import GGUFQuantizationConfig, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.utils import load_image


model_id = "Lightricks/LTX-2"
single_file_ckpt = "https://huggingface.co/unsloth/LTX-2-GGUF/blob/main/ltx-2-19b-dev-Q4_K_M.gguf"

device = "cuda:0"
dtype = torch.bfloat16
seed = 42

quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)

transformer = LTX2VideoTransformer3DModel.from_single_file(
    single_file_ckpt, config=model_id, subfolder="transformer", quantization_config=quantization_config
)

pipe = LTX2ImageToVideoPipeline.from_pretrained("Lightricks/LTX-2", transformer=transformer, torch_dtype=dtype)
pipe.enable_model_cpu_offload()

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
prompt = "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a breath-taking, movie-like shot."
negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."

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
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="./outputs/ltx2/i2v_gguf.mp4",
)
