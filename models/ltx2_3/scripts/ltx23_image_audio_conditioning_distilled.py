import os

import torch
from diffusers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
from diffusers.utils import load_image
from pipeline_ltx2_multimodal import LTX2AudioCondition, LTX2ImageCondition, LTX2MultiModalPipeline, load_audio
from sdnq import SDNQConfig  # noqa: F401 — registers SDNQ weight loader


DEVICE = "cuda:0"
DTYPE = torch.bfloat16

BASE_MODEL = "OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int8"
CONTROL_DOWNSCALE_FACTOR = 1

WIDTH = 960
HEIGHT = 544
SECONDS = 10
FRAME_RATE = 24.0
NUM_FRAMES = round(SECONDS * FRAME_RATE) // 8 * 8 + 1

PROMPT = """a cat is sitting in front of the camera and then speaks to the camera, moving its mouth speaks 'we should strive to make open source the frontier models for generating audio, video and images' then the cat stands and exists the scene"""
NEGATIVE_PROMPT = """blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."""

IMAGE_URL = "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/20260508111110.png"
AUDIO_URL = "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/man.mp3"

NUM_INFERENCE_STEPS = len(DISTILLED_SIGMA_VALUES)
GUIDANCE_SCALE = 1.0

SEED = None
# ----------------------------------------------------------------------------

if SEED is None:
    SEED = torch.randint(0, 2**32, (1,)).item()
    print(f"Using random seed: {SEED}")

transformer = LTX2VideoTransformer3DModel.from_pretrained(BASE_MODEL, subfolder="transformer", torch_dtype=DTYPE)
pipe = LTX2MultiModalPipeline.from_pretrained(BASE_MODEL, transformer=transformer, torch_dtype=DTYPE)

pipe.enable_group_offload(
    onload_device=torch.device(DEVICE),
    offload_type="leaf_level",
    use_stream=True,
    low_cpu_mem_usage=True,
)

waveform = load_audio(AUDIO_URL, target_sample_rate=pipe.audio_vae.config.sample_rate, seconds=SECONDS)
audio_conditions = [LTX2AudioCondition(audio=waveform, strength=1.0)]

image = load_image(IMAGE_URL).convert("RGB")
image_conditions = [LTX2ImageCondition(image=image, frame=0, strength=1.0)]

generator = torch.Generator(device="cpu").manual_seed(SEED)
result = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    width=WIDTH,
    height=HEIGHT,
    num_frames=NUM_FRAMES,
    frame_rate=FRAME_RATE,
    num_inference_steps=NUM_INFERENCE_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    audio_conditions=audio_conditions,
    image_conditions=image_conditions,
    control_downscale_factor=CONTROL_DOWNSCALE_FACTOR,
    control_strength=1.0,
    sigmas=DISTILLED_SIGMA_VALUES,
    use_cross_timestep=True,
    generator=generator,
    output_type="pil",
)

# 4. Save
os.makedirs("outputs/ltx23", exist_ok=True)
OUTPUT_PATH = f"outputs/ltx23/ltx23_image_audio_conditioning_distilled_{WIDTH}x{HEIGHT}_{SECONDS}s_seed{SEED}.mp4"
video, audio = result.frames[0], result.audio
encode_video(
    video,
    fps=FRAME_RATE,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path=OUTPUT_PATH,
)
print(f"Saved: {OUTPUT_PATH}")
