import gc
import os

import torch
from diffusers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2 import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.video_processor import VideoProcessor
from sdnq import SDNQConfig  # noqa: F401 — registers SDNQ into transformers
from transformers import Gemma3ForConditionalGeneration


device = "cuda:0"
dtype = torch.bfloat16
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
MODEL_PATH = "OzzyGT/LTX-2.3-Distilled"
SDNQ_BITS = 4  # 4 or 8
SDNQ_MODEL_PATH = f"OzzyGT/LTX-2.3-Distilled-sdnq-dynamic-int{SDNQ_BITS}"
ENCODE_CPU_OFFLOAD = False  # True = sequential cpu offload, False = keep text encoder on GPU
DENOISE_GROUP_OFFLOAD = True  # True = group offload transformer, False = keep transformer on GPU

width = 960
height = 544
seconds = 5
frame_rate = 24.0
seed = 42
num_frames = round(seconds * frame_rate) // 8 * 8 + 1

num_inference_steps = 8
guidance_scale = 1.0
audio_guidance_scale = 1.0
modality_scale = 3.0
audio_modality_scale = 3.0

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
    print(f"Using random seed: {seed}")
generator = torch.Generator(device="cpu").manual_seed(seed)

prompt = """A highly detailed macro cinematic shot inside a dense tropical rainforest just after heavy rain. Giant glossy leaves fill the frame, covered in crystal-clear water droplets that reflect the environment like tiny lenses. A bright metallic-blue butterfly rests on a leaf in the foreground, its wings slowly opening to reveal intricate shimmering patterns.
A sudden droplet falls from a higher leaf and lands nearby, causing smaller droplets to bounce and scatter in slow motion. The butterfly reacts, gently lifting off into the humid air. As it flutters away, the camera performs a subtle smooth push-in through layers of foliage, creating rich natural depth and parallax.
In the background, soft mist drifts between massive tree trunks while distant leaves sway slightly. Tiny floating pollen particles catch shafts of warm sunlight breaking through the canopy.
Ultra-realistic textures, natural lighting, shallow depth of field, cinematic focus transitions, physically accurate motion, rich environmental detail.
Sound description: soft rainforest ambience, distant birds, gentle water drips, subtle wing flutters."""


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ── Step 0: Encode prompts ────────────────────────────────────────────────────
print("Step 0: Encode prompts")

text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
    SDNQ_MODEL_PATH, subfolder="text_encoder", torch_dtype=dtype
)

embeds_pipe = LTX2Pipeline.from_pretrained(
    MODEL_PATH,
    text_encoder=text_encoder,
    transformer=None,
    vae=None,
    audio_vae=None,
    vocoder=None,
    scheduler=None,
    torch_dtype=dtype,
)
if ENCODE_CPU_OFFLOAD:
    embeds_pipe.enable_sequential_cpu_offload()
else:
    embeds_pipe.text_encoder = embeds_pipe.text_encoder.to(device)
    embeds_pipe.connectors = embeds_pipe.connectors.to(device)

with torch.inference_mode():
    prompt_embeds, prompt_attention_mask, *_ = embeds_pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=False,
    )

prompt_embeds = prompt_embeds.cpu()
prompt_attention_mask = prompt_attention_mask.cpu()

del embeds_pipe, text_encoder
flush()

# ── Step 1: Denoise ───────────────────────────────────────────────────────────
print(f"Step 1: Denoise {width}x{height}")

transformer = LTX2VideoTransformer3DModel.from_pretrained(
    SDNQ_MODEL_PATH, subfolder="transformer", torch_dtype=dtype, device_map="cpu"
)

pipe = LTX2Pipeline.from_pretrained(
    MODEL_PATH,
    transformer=transformer,
    text_encoder=None,
    tokenizer=None,
    torch_dtype=dtype,
)
if DENOISE_GROUP_OFFLOAD:
    pipe.enable_group_offload(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="leaf_level",
        low_cpu_mem_usage=True,
        exclude_modules=["vae", "audio_vae", "vocoder"],
    )
else:
    pipe.transformer.to(device)

video_latent, audio_latent = pipe(
    prompt_embeds=prompt_embeds.to(device=device, dtype=dtype),
    prompt_attention_mask=prompt_attention_mask.to(device=device),
    guidance_scale=guidance_scale,
    audio_guidance_scale=audio_guidance_scale,
    width=width,
    height=height,
    num_frames=num_frames,
    frame_rate=frame_rate,
    num_inference_steps=num_inference_steps,
    modality_scale=modality_scale,
    audio_modality_scale=audio_modality_scale,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

video_latent = video_latent.cpu()
audio_latent = audio_latent.cpu()

vae = pipe.vae
audio_vae = pipe.audio_vae
vocoder = pipe.vocoder
audio_sample_rate = vocoder.config.output_sampling_rate

del pipe, transformer, prompt_embeds, prompt_attention_mask
flush()

# ── Step 2: Decode ────────────────────────────────────────────────────────────
print("Step 2: Decode video + audio")

vae.to(device)
vae.enable_tiling()
with torch.no_grad():
    video = vae.decode(video_latent.to(device=device, dtype=vae.dtype), return_dict=False)[0]

video_processor = VideoProcessor(vae_scale_factor=vae.spatial_compression_ratio)
video = video_processor.postprocess_video(video, output_type="np")

del vae, video_latent
flush()

audio_vae.to(device)
vocoder.to(device)
with torch.no_grad():
    audio_latent_gpu = audio_latent.to(device=device, dtype=audio_vae.dtype)
    mel = audio_vae.decode(audio_latent_gpu, return_dict=False)[0]
    audio = vocoder(mel)
audio = audio.cpu()

del audio_vae, vocoder, audio_latent, audio_latent_gpu, mel
flush()

# ── Save ──────────────────────────────────────────────────────────────────────
print("Save output")

if not os.path.exists("./outputs/ltx23"):
    os.makedirs("./outputs/ltx23")

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=audio_sample_rate,
    output_path="./outputs/ltx23/t2v_sdnq_one_stage_distilled_steps.mp4",
)
