import gc
import os

import torch
from diffusers import LTX2ImageToVideoPipeline, LTX2LatentUpsamplePipeline, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import load_image
from transformers import Gemma3ForConditionalGeneration


torch_dtype = torch.bfloat16
device = "cuda"
model_path = "Lightricks/LTX-2"
width = 384
height = 256
num_frames = 241
generator = torch.Generator("cuda").manual_seed(42)

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)

text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
    "OzzyGT/LTX-2-bnb-4bit-text-encoder",
    dtype=torch_dtype,
    device_map="cpu",
)

transformer = LTX2VideoTransformer3DModel.from_pretrained(
    "OzzyGT/LTX-2-bnb-4bit-transformer",
    torch_dtype=torch_dtype,
    device_map="cpu",
)

pipe = LTX2ImageToVideoPipeline.from_pretrained(
    model_path, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
)
pipe.vae.enable_tiling(
    tile_sample_min_height=256,
    tile_sample_min_width=256,
    tile_sample_min_num_frames=16,
    tile_sample_stride_height=192,
    tile_sample_stride_width=192,
    tile_sample_stride_num_frames=8,
)
pipe.vae.use_framewise_encoding = True
pipe.vae.use_framewise_decoding = True
pipe.load_lora_weights(
    "Lightricks/LTX-2", adapter_name="stage_2_distilled", weight_name="ltx-2-19b-distilled-lora-384.safetensors"
)
pipe.enable_model_cpu_offload()

prompt = "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling apart in gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, floating in slow arcs before settling back onto the ground. The astronaut pushes free in a deliberate, weightless motion, small fragments of the egg tumbling and spinning through the air. In the background, the deep darkness of space subtly shifts as stars glide with the camera's movement, emphasizing vast depth and scale. The camera performs a smooth, cinematic slow push-in, with natural parallax between the foreground dust, the astronaut, and the distant starfield. Ultra-realistic detail, physically accurate low-gravity motion, cinematic lighting, and a breath-taking, movie-like shot."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

frame_rate = 24.0
video_latent, audio_latent = pipe(
    prompt=prompt,
    image=image,
    width=width,
    height=height,
    num_frames=num_frames,
    frame_rate=frame_rate,
    num_inference_steps=8,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "rootonchair/LTX-2-19b-distilled",
    subfolder="latent_upsampler",
    torch_dtype=torch_dtype,
)
upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=device)
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

latent_upsampler.to("cpu")
del video_latent
del upsample_pipe
del latent_upsampler
gc.collect()
torch.cuda.empty_cache()

video, audio = pipe(
    image=image,
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width * 2,
    height=height * 2,
    num_frames=num_frames,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    generator=generator,
    guidance_scale=1.0,
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
    output_path="./outputs/ltx2/i2v_2_stages_distilled_lora_bnb_4bit.mp4",
)
