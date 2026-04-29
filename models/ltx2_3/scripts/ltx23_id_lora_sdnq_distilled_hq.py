import gc
import os
from io import BytesIO

import requests
import torch
from diffusers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from PIL import Image
from pipeline_ltx2_multimodal import LTX2ImageCondition, LTX2MultiModalPipeline, load_audio
from sdnq import SDNQConfig  # noqa: F401 — registers SDNQ weight loader


DEVICE = "cuda:0"
DTYPE = torch.bfloat16

BASE_MODEL = "OzzyGT/LTX-2.3-sdnq-dynamic-int8"
ID_LORA_REPO = "AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K"
ID_LORA_WEIGHT = "lora_weights.safetensors"
DISTILLED_LORA_REPO = "Lightricks/LTX-2.3"
DISTILLED_LORA_WEIGHT = "ltx-2.3-22b-distilled-lora-384.safetensors"
UPSAMPLER_PATH = "OzzyGT/LTX-2.3-upsampler-x2"

# Demo uses just distilled with different strengths
DISTILLED_STRENGTH_STAGE_1 = 0.25
DISTILLED_STRENGTH_STAGE_2 = 0.5
ID_LORA_STRENGTH_STAGE_1 = 1.0

FIRST_FRAME_URL = "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/id_lora_first_frame.png"
REFERENCE_AUDIO_URL = (
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/id_lora_reference.wav"
)

WIDTH_STAGE_1 = 512
HEIGHT_STAGE_1 = 512
WIDTH_STAGE_2 = WIDTH_STAGE_1 * 2
HEIGHT_STAGE_2 = HEIGHT_STAGE_1 * 2
SECONDS = 5
FRAME_RATE = 24.0
NUM_FRAMES = round(SECONDS * FRAME_RATE) // 8 * 8 + 1  # 8n+1

# Tag-format prompt the ID-LoRA was trained on — [VISUAL]/[SPEECH]/[SOUNDS] sections.
PROMPT = """[VISUAL]: A medium shot features a young man with medium-length, curly brown hair and light blue eyes, sitting on a beige or light brown couch. He is wearing a light blue button-up shirt and a red and white patterned tie. His mouth is slightly open as he speaks or reacts. In the background, there is a blurry room setting with warm lighting.
[SPEECH]: We are proud to introduce ID-LoRA.
[SOUNDS]: The speaker has a moderate volume and a conversational tone, sounding engaged and natural. They are close to the microphone. Light, instrumental background music plays softly, creating a calm atmosphere."""

NEGATIVE_PROMPT = """blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."""

# Stage 1
NUM_INFERENCE_STEPS_STAGE_1 = 15
GUIDANCE_SCALE = 3.0
AUDIO_GUIDANCE_SCALE = 7.0
GUIDANCE_RESCALE = 0.45
AUDIO_GUIDANCE_RESCALE = 1.0
STG_SCALE = 1.0
AUDIO_STG_SCALE = 1.0
STG_BLOCKS = [28]
MODALITY_SCALE = 3.0
AUDIO_MODALITY_SCALE = 3.0
IDENTITY_GUIDANCE_SCALE = 3.0

# Stage 2
NUM_INFERENCE_STEPS_STAGE_2 = 3

SEED = 42
# ----------------------------------------------------------------------------

if SEED is None:
    SEED = torch.randint(0, 2**32, (1,)).item()
    print(f"Using random seed: {SEED}")

# Single pipeline shared across both stages — we just toggle adapter weights between calls.
transformer = LTX2VideoTransformer3DModel.from_pretrained(BASE_MODEL, subfolder="transformer", torch_dtype=DTYPE)
pipe = LTX2MultiModalPipeline.from_pretrained(BASE_MODEL, transformer=transformer, torch_dtype=DTYPE)
pipe.load_lora_weights(ID_LORA_REPO, weight_name=ID_LORA_WEIGHT, adapter_name="id_lora")
pipe.load_lora_weights(DISTILLED_LORA_REPO, weight_name=DISTILLED_LORA_WEIGHT, adapter_name="distilled")
pipe.enable_group_offload(
    onload_device=torch.device(DEVICE),
    offload_type="leaf_level",
    use_stream=True,
    low_cpu_mem_usage=True,
)

first_frame = Image.open(BytesIO(requests.get(FIRST_FRAME_URL, timeout=60).content)).convert("RGB")
image_conditions = [LTX2ImageCondition(image=first_frame, frame=0, strength=1.0)]

reference_waveform = load_audio(
    REFERENCE_AUDIO_URL,
    target_sample_rate=pipe.audio_vae.config.sample_rate,
    seconds=SECONDS,
)

# ─── Stage 1 ─────────────────────────────────────────────────────────────────
pipe.set_adapters(["id_lora", "distilled"], adapter_weights=[ID_LORA_STRENGTH_STAGE_1, DISTILLED_STRENGTH_STAGE_1])

generator = torch.Generator(device="cpu").manual_seed(SEED)
stage1_video_latents, stage1_audio_latents = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    width=WIDTH_STAGE_1,
    height=HEIGHT_STAGE_1,
    num_frames=NUM_FRAMES,
    frame_rate=FRAME_RATE,
    num_inference_steps=NUM_INFERENCE_STEPS_STAGE_1,
    guidance_scale=GUIDANCE_SCALE,
    audio_guidance_scale=AUDIO_GUIDANCE_SCALE,
    guidance_rescale=GUIDANCE_RESCALE,
    audio_guidance_rescale=AUDIO_GUIDANCE_RESCALE,
    stg_scale=STG_SCALE,
    audio_stg_scale=AUDIO_STG_SCALE,
    spatio_temporal_guidance_blocks=STG_BLOCKS,
    modality_scale=MODALITY_SCALE,
    audio_modality_scale=AUDIO_MODALITY_SCALE,
    image_conditions=image_conditions,
    control_audio=reference_waveform,
    control_audio_strength=1.0,
    identity_guidance_scale=IDENTITY_GUIDANCE_SCALE,
    generator=generator,
    output_type="latent",
    return_dict=False,
)
print(f"Stage 1 video latents: {stage1_video_latents.shape}")
print(f"Stage 1 audio latents: {stage1_audio_latents.shape}")

# ─── Spatial upsample (2x) ───────────────────────────────────────────────────
latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(UPSAMPLER_PATH, torch_dtype=DTYPE)
latent_upsampler.to(DEVICE)
with torch.no_grad():
    upscaled_video_latents = latent_upsampler(stage1_video_latents.to(device=DEVICE, dtype=DTYPE))
print(f"Upscaled video latents: {upscaled_video_latents.shape}")

del stage1_video_latents, latent_upsampler
gc.collect()
torch.cuda.empty_cache()

# ─── Stage 2 ─────────────────────────────────────────────────────────────────
pipe.set_adapters(["distilled"], adapter_weights=[DISTILLED_STRENGTH_STAGE_2])

stage2_generator = torch.Generator(device="cpu").manual_seed(SEED)
video, _ = pipe(
    prompt=PROMPT,
    width=WIDTH_STAGE_2,
    height=HEIGHT_STAGE_2,
    num_frames=NUM_FRAMES,
    frame_rate=FRAME_RATE,
    num_inference_steps=NUM_INFERENCE_STEPS_STAGE_2,
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
    guidance_scale=1.0,
    image_conditions=image_conditions,
    latents=upscaled_video_latents.to(device=DEVICE, dtype=DTYPE),
    audio_latents=stage1_audio_latents.to(device=DEVICE, dtype=DTYPE),
    generator=stage2_generator,
    output_type="pil",
    return_dict=False,
)

# Decode stage-1 audio directly — this is what ID-LoRA's HQ uses (audio frozen across stage 2).
with torch.no_grad():
    stage1_audio_latents_dev = stage1_audio_latents.to(device=DEVICE, dtype=pipe.audio_vae.dtype)
    mel = pipe.audio_vae.decode(stage1_audio_latents_dev, return_dict=False)[0]
    with torch.autocast(device_type=mel.device.type, dtype=torch.float32):
        audio = pipe.vocoder(mel)

# Save
os.makedirs("outputs/ltx23", exist_ok=True)
OUTPUT_PATH = f"outputs/ltx23/ltx2_id_lora_distilled_hq_{WIDTH_STAGE_2}x{HEIGHT_STAGE_2}_{SECONDS}s_seed{SEED}.mp4"
encode_video(
    video[0],
    fps=FRAME_RATE,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path=OUTPUT_PATH,
)
print(f"Saved: {OUTPUT_PATH}")
