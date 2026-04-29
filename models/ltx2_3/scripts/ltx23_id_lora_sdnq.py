import os
from io import BytesIO

import requests
import torch
from diffusers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from PIL import Image
from pipeline_ltx2_multimodal import LTX2ImageCondition, LTX2MultiModalPipeline, load_audio
from sdnq import SDNQConfig  # noqa: F401 — registers SDNQ weight loader


DEVICE = "cuda:0"
DTYPE = torch.bfloat16

BASE_MODEL = "OzzyGT/LTX-2.3-sdnq-dynamic-int8"
ID_LORA_REPO = "AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K"
ID_LORA_WEIGHT = "lora_weights.safetensors"

FIRST_FRAME_URL = "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/id_lora_first_frame.png"
REFERENCE_AUDIO_URL = (
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/id_lora_reference.wav"
)

WIDTH = 512
HEIGHT = 512
SECONDS = 5
FRAME_RATE = 24.0
NUM_FRAMES = round(SECONDS * FRAME_RATE) // 8 * 8 + 1  # 8n+1

# Tag-format prompt the ID-LoRA was trained on — [VISUAL]/[SPEECH]/[SOUNDS] sections.
PROMPT = """[VISUAL]: A medium shot features a young man with medium-length, curly brown hair and light blue eyes, sitting on a beige or light brown couch. He is wearing a light blue button-up shirt and a red and white patterned tie. His mouth is slightly open as he speaks or reacts. In the background, there is a blurry room setting with warm lighting.
[SPEECH]: We are proud to introduce ID-LoRA.
[SOUNDS]: The speaker has a moderate volume and a conversational tone, sounding engaged and natural. They are close to the microphone. Light, instrumental background music plays softly, creating a calm atmosphere."""

NEGATIVE_PROMPT = """blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."""

NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 3.0
AUDIO_GUIDANCE_SCALE = 7.0
GUIDANCE_RESCALE = 0.7
AUDIO_GUIDANCE_RESCALE = 0.7
STG_SCALE = 1.0
AUDIO_STG_SCALE = 1.0
STG_BLOCKS = [29]
MODALITY_SCALE = 3.0
AUDIO_MODALITY_SCALE = 3.0
IDENTITY_GUIDANCE_SCALE = 3.0  # opt-in extra forward pass per step — ID-LoRA-specific

SEED = 42
# ----------------------------------------------------------------------------

if SEED is None:
    SEED = torch.randint(0, 2**32, (1,)).item()
    print(f"Using random seed: {SEED}")

transformer = LTX2VideoTransformer3DModel.from_pretrained(BASE_MODEL, subfolder="transformer", torch_dtype=DTYPE)
pipe = LTX2MultiModalPipeline.from_pretrained(BASE_MODEL, transformer=transformer, torch_dtype=DTYPE)
pipe.load_lora_weights(ID_LORA_REPO, weight_name=ID_LORA_WEIGHT)
pipe.enable_group_offload(
    onload_device=torch.device(DEVICE),
    offload_type="leaf_level",
    use_stream=True,
    low_cpu_mem_usage=True,
)

# First-frame image conditioning (face anchor at frame 0 = replace condition)
first_frame = Image.open(BytesIO(requests.get(FIRST_FRAME_URL, timeout=60).content)).convert("RGB")
image_conditions = [LTX2ImageCondition(image=first_frame, frame=0, strength=1.0)]

# Reference audio — the speaker identity / voice timbre to transfer onto the generated audio
reference_waveform = load_audio(
    REFERENCE_AUDIO_URL,
    target_sample_rate=pipe.audio_vae.config.sample_rate,
    seconds=SECONDS,
)

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
    audio_guidance_scale=AUDIO_GUIDANCE_SCALE,
    guidance_rescale=GUIDANCE_RESCALE,
    audio_guidance_rescale=AUDIO_GUIDANCE_RESCALE,
    stg_scale=STG_SCALE,
    audio_stg_scale=AUDIO_STG_SCALE,
    spatio_temporal_guidance_blocks=STG_BLOCKS,
    modality_scale=MODALITY_SCALE,
    audio_modality_scale=AUDIO_MODALITY_SCALE,
    # First-frame face anchor
    image_conditions=image_conditions,
    # Reference voice → (audio IC-LoRA)
    control_audio=reference_waveform,
    control_audio_strength=1.0,
    # Opt-in extra forward pass per step that amplifies the reference's contribution.
    # ID-LoRA was trained for this; vanilla audio LoRAs should leave it at 0.0.
    identity_guidance_scale=IDENTITY_GUIDANCE_SCALE,
    generator=generator,
    output_type="pil",
)

# Save
os.makedirs("outputs/ltx23", exist_ok=True)
OUTPUT_PATH = f"outputs/ltx23/ltx2_id_lora_{WIDTH}x{HEIGHT}_{SECONDS}s_seed{SEED}.mp4"
video, audio = result.frames[0], result.audio
encode_video(
    video,
    fps=FRAME_RATE,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path=OUTPUT_PATH,
)
print(f"Saved: {OUTPUT_PATH}")
