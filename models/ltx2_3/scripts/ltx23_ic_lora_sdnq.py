import os

import torch
from diffusers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from pipeline_ltx2_multimodal import LTX2AudioCondition, LTX2MultiModalPipeline, load_audio, load_video
from sdnq import SDNQConfig  # noqa: F401 — registers SDNQ weight loader


DEVICE = "cuda:0"
DTYPE = torch.bfloat16

BASE_MODEL = "OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int8"
IC_LORA_REPO = "oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint"
CONTROL_DOWNSCALE_FACTOR = 1

AUDIO_PATH = "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/ic_Lora_source_audio.mp3"
SOURCE_VIDEO_PATH = (
    "https://huggingface.co/datasets/OzzyGT/diffusers-examples/resolve/main/ltx2/source_video_f23eecf5ca74.mp4"
)

WIDTH = 960
HEIGHT = 544
SECONDS = 5
FRAME_RATE = 24.0
NUM_FRAMES = round(SECONDS * FRAME_RATE) // 8 * 8 + 1

PROMPT = """A handcrafted felt cutout cinematic diorama at soft golden morning light reveals a chubby felt capybara composed of warm tan and light brown felt layers, its rounded body softly stitched with visible seams and fuzzy texture. Its small embroidered black bead eyes and tiny stitched nose give it a calm, slightly curious expression. The capybara sits beside a gentle felt river made of rippling blue fabric, with layered reeds and lily pads subtly swaying. It wears a tiny moss-green felt satchel with delicate stitched straps resting against its side.
As the camera performs a slow, smooth push-in from a low angle, keeping the capybara centered in the midground, a faint breeze causes nearby felt grasses and flowers to softly flutter. The capybara blinks once, tilts its head slightly to the side, and its stitched mouth opens just enough to speak. With a soft, puzzled tone, it says: "wait... what?... is this diffusers with an IC-Lora?"
Warm dappled sunlight creates gentle shadows between the layered felt elements, highlighting the tactile fibers, stitched edges, and handcrafted imperfections. Subtle ambient audio of soft water movement and distant birds enhances the calm, whimsical atmosphere."""
NEGATIVE_PROMPT = """blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."""

NUM_INFERENCE_STEPS = 8
GUIDANCE_SCALE = 1.0
AUDIO_GUIDANCE_SCALE = 1.0
STG_SCALE = 0.0
MODALITY_SCALE = 3.0
AUDIO_MODALITY_SCALE = 3.0

SEED = None
# ----------------------------------------------------------------------------

if SEED is None:
    SEED = torch.randint(0, 2**32, (1,)).item()
    print(f"Using random seed: {SEED}")

transformer = LTX2VideoTransformer3DModel.from_pretrained(BASE_MODEL, subfolder="transformer", torch_dtype=DTYPE)
pipe = LTX2MultiModalPipeline.from_pretrained(BASE_MODEL, transformer=transformer, torch_dtype=DTYPE)
pipe.load_lora_weights(IC_LORA_REPO)
pipe.enable_group_offload(
    onload_device=torch.device(DEVICE),
    offload_type="leaf_level",
    use_stream=True,
    low_cpu_mem_usage=True,
)

waveform = load_audio(AUDIO_PATH, target_sample_rate=pipe.audio_vae.config.sample_rate, seconds=SECONDS)
audio_conditions = [LTX2AudioCondition(audio=waveform, strength=1.0)]
control_frames = load_video(SOURCE_VIDEO_PATH)[:NUM_FRAMES]

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
    stg_scale=STG_SCALE,
    audio_stg_scale=STG_SCALE,
    modality_scale=MODALITY_SCALE,
    audio_modality_scale=AUDIO_MODALITY_SCALE,
    # Audio drives the video
    audio_conditions=audio_conditions,
    # Source video guides spatial outpainting via the IC-LoRA (sequence-concat)
    control_video=control_frames,
    control_downscale_factor=CONTROL_DOWNSCALE_FACTOR,
    control_strength=1.0,
    generator=generator,
    output_type="pil",
)

# 4. Save
os.makedirs("outputs/ltx23", exist_ok=True)
OUTPUT_PATH = f"outputs/ltx23/ltx2_ic_lora_outpaint_{WIDTH}x{HEIGHT}_{SECONDS}s_seed{SEED}.mp4"
video, audio = result.frames[0], result.audio
encode_video(
    video,
    fps=FRAME_RATE,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path=OUTPUT_PATH,
)
print(f"Saved: {OUTPUT_PATH}")
