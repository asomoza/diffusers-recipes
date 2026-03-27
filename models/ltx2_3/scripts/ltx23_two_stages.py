"""
LTX 2.3 Two-Stage Text-to-Video Inference Script

Pipeline:
  Encode  -> Encode prompts (text_encoder + connectors only)
  Stage 1 -> Generate video+audio at reduced resolution with CFG (non-distilled, 30 steps)
  Upscale -> Spatially upscale latents (2x or 1.5x)
  Stage 2 -> Refine at full resolution with distilled model (no CFG, 3 steps)
  Decode  -> Streaming temporal decode + audio decode

Each step loads only what it needs and frees everything before the next step to minimize VRAM usage.

Benchmarks (1536x1024, 10s @ 24fps, 30 Stage 1 steps + 3 Stage 2 steps, RTX 5090):
  All values are true peaks measured by a background polling thread (100ms interval).
  VRAM is measured via CUDA driver (baseline-subtracted). RAM excludes mmap'd file pages.

  UPSCALE_FACTOR=2, streaming decode:
  ┌────────────────────────┬──────────┬───────────┬──────────┐
  │ Step                   │ Time     │ Peak VRAM │ Peak RAM │
  ├────────────────────────┼──────────┼───────────┼──────────┤
  │ Encode prompts         │     9.8s │   5.01 GB │ 24.87 GB │
  │ Stage 1 (768x512)      │   8m 39s │   9.93 GB │  2.33 GB │
  │ Spatial Upscale 2x     │     1.3s │   2.46 GB │  2.40 GB │
  │ Stage 2 (1536x1024)    │   1m 11s │  13.60 GB │  2.43 GB │
  │ Decode (streaming)     │    18.4s │   5.12 GB │ 12.27 GB │
  │ TOTAL                  │  10m 28s │  13.60 GB │ 24.87 GB │
  └────────────────────────┴──────────┴───────────┴──────────┘

  Note: The 24.87 GB RAM peak is transient — it occurs while loading the Gemma3 12B
  text encoder and drops to <5 GB during inference. This fits comfortably in 32 GB RAM.
"""

import ctypes
import gc
import os
import threading
import time

import psutil
import torch

from diffusers import AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.video_processor import VideoProcessor


# ── Configuration ─────────────────────────────────────────────────────────────
device = "cuda:0"
offload_device = "cpu"
dtype = torch.bfloat16

STAGE_1_MODEL_PATH = "OzzyGT/LTX-2.3"
DISTILLED_MODEL_PATH = "OzzyGT/LTX-2.3-Distilled"
UPSCALE_FACTOR = 2  # 2 or 1.5
UPSAMPLER_PATH = "OzzyGT/LTX-2.3-upsampler-x2" if UPSCALE_FACTOR == 2 else "OzzyGT/LTX-2.3-upsampler-x1.5"
DECODE_MODE = "streaming"  # "streaming" = low VRAM decode (tiles on CPU), "tiling" = on-GPU spatial tiling
LOW_CPU_MEM_USAGE = True  # Reduces RAM usage during group offload by loading weights lazily

# Target resolution (Stage 2 output). Stage 1 runs at reduced resolution.
width = 1536
height = 1024
seconds = 10
frame_rate = 24.0
seed = None
# Frame count must be k*8+1 for VAE temporal alignment
num_frames = round(seconds * frame_rate) // 8 * 8 + 1

# Stage 1 resolution (derived from target and upscale factor)
stage1_width = round(width / UPSCALE_FACTOR)
stage1_height = round(height / UPSCALE_FACTOR)

# Stage 1 pipeline args (non-distilled, with CFG)
num_inference_steps = 30
guidance_scale = 3.0
audio_guidance_scale = 7.0
guidance_rescale = 0.7
audio_guidance_rescale = 0.7
stg_scale = 1.0
audio_stg_scale = 1.0
spatio_temporal_guidance_blocks = [28]
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

negative_prompt = """ blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions,
  unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent
  perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color
  banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction,
  mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery
  movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized
  filters, or AI artifacts.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def get_ram_gb():
    """Return heap RAM in GB (excludes mmap'd file pages that the OS can reclaim)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) / 1024**2
    except FileNotFoundError:
        pass
    return psutil.Process().memory_full_info().uss / 1024**3


def get_gpu_used_gb():
    """Total GPU memory used by this process (CUDA driver level, not just PyTorch)."""
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1024**3


class PeakMemoryMonitor:
    """Background thread that polls process RSS and GPU memory to capture true peaks."""

    def __init__(self, interval=0.1):
        self.peak_ram = 0.0
        self.peak_vram = 0.0
        self._running = False
        self._thread = None
        self._interval = interval

    def start(self):
        self.peak_ram = get_ram_gb()
        self.peak_vram = get_gpu_used_gb()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            self.peak_ram = max(self.peak_ram, get_ram_gb())
            self.peak_vram = max(self.peak_vram, get_gpu_used_gb())
            time.sleep(self._interval)

    def stop(self):
        self._running = False
        self._thread.join()
        return self.peak_vram, self.peak_ram


monitor = PeakMemoryMonitor(interval=0.1)


def log_step(step_name, start_time):
    elapsed = time.time() - start_time
    peak_vram_abs, peak_ram = monitor.stop()
    peak_vram = peak_vram_abs - vram_baseline
    print(f"  [{step_name}] {elapsed:.1f}s | Peak VRAM: {peak_vram:.2f} GB | Peak RAM: {peak_ram:.2f} GB")
    return peak_vram, peak_ram


# Measure VRAM baseline before anything is loaded (CUDA context, other processes, etc.)
# All VRAM measurements will be relative to this baseline.
torch.cuda.init()
vram_baseline = get_gpu_used_gb()
print(f"VRAM baseline: {vram_baseline:.2f} GB")

script_start = time.time()
global_peak_vram = 0.0
global_peak_ram = 0.0


def step_start(step_name):
    monitor.start()
    print(f"\n{'─' * 70}")
    print(f"  {step_name}")
    print(f"{'─' * 70}")
    return time.time()


def step_end(step_name, t0):
    global global_peak_vram, global_peak_ram
    peak_vram, peak_ram = log_step(step_name, t0)
    global_peak_vram = max(global_peak_vram, peak_vram)
    global_peak_ram = max(global_peak_ram, peak_ram)


def vae_temporal_decode_streaming(
    vae: AutoencoderKLLTX2Video,
    latents_cpu: torch.Tensor,
    *,
    decode_device: torch.device,
    temb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode long video latents with lower peak VRAM by streaming temporal tiles to GPU.

    Each tile is decoded independently and moved back to CPU before the next tile is processed,
    so peak GPU memory is proportional to a single tile rather than the entire video.
    """
    tile_latent_min_num_frames = vae.tile_sample_min_num_frames // vae.temporal_compression_ratio
    if latents_cpu.shape[2] <= tile_latent_min_num_frames:
        latents = latents_cpu.to(device=decode_device, dtype=vae.dtype, non_blocking=True)
        return vae.decode(latents, temb=temb, return_dict=False)[0].cpu()

    num_frames = latents_cpu.shape[2]
    num_sample_frames = (num_frames - 1) * vae.temporal_compression_ratio + 1

    tile_latent_stride_num_frames = vae.tile_sample_stride_num_frames // vae.temporal_compression_ratio
    blend_num_frames = vae.tile_sample_min_num_frames - vae.tile_sample_stride_num_frames

    result_tiles: list[torch.Tensor] = []
    prev_row_tile: torch.Tensor | None = None

    for i in range(0, num_frames, tile_latent_stride_num_frames):
        tile_cpu = latents_cpu[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
        tile = tile_cpu.to(device=decode_device, dtype=vae.dtype, non_blocking=True)

        # Temporarily disable framewise decoding to prevent the VAE from
        # re-entering _temporal_tiled_decode on the already-sliced tile.
        saved_framewise = vae.use_framewise_decoding
        vae.use_framewise_decoding = False
        decoded = vae.decode(tile, temb=temb, return_dict=False)[0]
        vae.use_framewise_decoding = saved_framewise
        row_tile = decoded.cpu()

        if i > 0:
            row_tile = row_tile[:, :, :-1, :, :]

        if prev_row_tile is None:
            result_tiles.append(row_tile[:, :, : vae.tile_sample_stride_num_frames + 1, :, :])
        else:
            stitched = vae.blend_t(prev_row_tile, row_tile, blend_num_frames)
            stitched = stitched[:, :, : vae.tile_sample_stride_num_frames, :, :]
            result_tiles.append(stitched)

        prev_row_tile = row_tile
        del tile, decoded

    return torch.cat(result_tiles, dim=2)[:, :, :num_sample_frames]


# ──────────────────────────────────────────────────────────────────────────────
# Step 0: Encode prompts (text_encoder + connectors only, no transformer/VAE)
# ──────────────────────────────────────────────────────────────────────────────
t0 = step_start("Step 0: Encode prompts")
embeds_pipe = LTX2Pipeline.from_pretrained(
    STAGE_1_MODEL_PATH,
    transformer=None,
    vae=None,
    audio_vae=None,
    vocoder=None,
    scheduler=None,
    torch_dtype=dtype,
)
embeds_pipe.enable_group_offload(
    onload_device=torch.device(device), offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=LOW_CPU_MEM_USAGE
)

with torch.inference_mode():
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
        embeds_pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
        )
    )

prompt_embeds = prompt_embeds.to(offload_device)
prompt_attention_mask = prompt_attention_mask.to(offload_device)
negative_prompt_embeds = negative_prompt_embeds.to(offload_device)
negative_prompt_attention_mask = negative_prompt_attention_mask.to(offload_device)
print(f"  prompt_embeds: {prompt_embeds.shape}")

del embeds_pipe
flush()
step_end("Step 0: Encode prompts", t0)

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Generate at reduced resolution (no text_encoder needed)
# ──────────────────────────────────────────────────────────────────────────────
t0 = step_start(f"Stage 1: Generate at {stage1_width}x{stage1_height}")
pipe = LTX2Pipeline.from_pretrained(
    STAGE_1_MODEL_PATH,
    text_encoder=None,
    tokenizer=None,
    torch_dtype=dtype,
)
pipe.enable_group_offload(
    onload_device=torch.device(device), offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=LOW_CPU_MEM_USAGE
)

video_latent, audio_latent = pipe(
    prompt_embeds=prompt_embeds.to(device=device, dtype=dtype),
    prompt_attention_mask=prompt_attention_mask.to(device=device),
    negative_prompt_embeds=negative_prompt_embeds.to(device=device, dtype=dtype),
    negative_prompt_attention_mask=negative_prompt_attention_mask.to(device=device),
    width=stage1_width,
    height=stage1_height,
    num_frames=num_frames,
    frame_rate=frame_rate,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    audio_guidance_scale=audio_guidance_scale,
    guidance_rescale=guidance_rescale,
    audio_guidance_rescale=audio_guidance_rescale,
    stg_scale=stg_scale,
    audio_stg_scale=audio_stg_scale,
    spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
    modality_scale=modality_scale,
    audio_modality_scale=audio_modality_scale,
    generator=generator,
    output_type="latent",
    return_dict=False,
)
print(f"  Video latent: {video_latent.shape}")
print(f"  Audio latent: {audio_latent.shape}")

video_latent = video_latent.to(offload_device)
audio_latent = audio_latent.to(offload_device)
del pipe
flush()
step_end(f"Stage 1: Generate at {stage1_width}x{stage1_height}", t0)

# ──────────────────────────────────────────────────────────────────────────────
# Spatial Upscale: latent upsampling
# ──────────────────────────────────────────────────────────────────────────────
t0 = step_start(f"Spatial Upscale: {UPSCALE_FACTOR}x")
latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(UPSAMPLER_PATH, torch_dtype=dtype)
latent_upsampler.to(device)

with torch.no_grad():
    upscaled_video_latent = latent_upsampler(video_latent.to(device=device, dtype=dtype))
print(f"  Upscaled video latent: {upscaled_video_latent.shape}")

upscaled_video_latent = upscaled_video_latent.to(offload_device)
del video_latent, latent_upsampler
flush()
step_end(f"Spatial Upscale: {UPSCALE_FACTOR}x", t0)

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Refine at full resolution with distilled model (no text_encoder)
# Output latents — we decode separately with streaming temporal tiling
# ──────────────────────────────────────────────────────────────────────────────
t0 = step_start(f"Stage 2: Refine at {width}x{height}")
pipe_stage2 = LTX2Pipeline.from_pretrained(
    DISTILLED_MODEL_PATH,
    text_encoder=None,
    tokenizer=None,
    torch_dtype=dtype,
)
pipe_stage2.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe_stage2.scheduler.config, use_dynamic_shifting=False, shift_terminal=None
)
pipe_stage2.enable_group_offload(
    onload_device=torch.device(device), offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=LOW_CPU_MEM_USAGE
)

video_latent, audio_latent = pipe_stage2(
    prompt_embeds=prompt_embeds.to(device=device, dtype=dtype),
    prompt_attention_mask=prompt_attention_mask.to(device=device),
    latents=upscaled_video_latent.to(device=device, dtype=dtype),
    audio_latents=audio_latent.to(device=device, dtype=dtype),
    width=width,
    height=height,
    num_frames=num_frames,
    frame_rate=frame_rate,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)
print(f"  Stage 2 video latent: {video_latent.shape}")
print(f"  Stage 2 audio latent: {audio_latent.shape}")

video_latent = video_latent.to(offload_device)
audio_latent = audio_latent.to(offload_device)

# Keep references to VAE components for decode, free the rest
vae = pipe_stage2.vae
audio_vae = pipe_stage2.audio_vae
vocoder = pipe_stage2.vocoder
audio_sample_rate = vocoder.config.output_sampling_rate

del prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
del upscaled_video_latent, pipe_stage2
flush()
step_end(f"Stage 2: Refine at {width}x{height}", t0)

# ──────────────────────────────────────────────────────────────────────────────
# Decode: Video + Audio
# ──────────────────────────────────────────────────────────────────────────────
t0 = step_start(f"Decode: Video ({DECODE_MODE}) + Audio")
vae.to(device)

if DECODE_MODE == "streaming":
    # Streaming temporal decode: one tile at a time on GPU, blend on CPU.
    # No spatial tiling — avoids grid artifacts at high resolutions.
    with torch.no_grad():
        video = vae_temporal_decode_streaming(vae, video_latent, decode_device=torch.device(device))
else:
    # Tiling decode: full video on GPU with spatial+temporal tiling.
    # Higher VRAM but faster. May show grid artifacts at high resolutions.
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
step_end(f"Decode: Video ({DECODE_MODE}) + Audio", t0)

# ──────────────────────────────────────────────────────────────────────────────
# Save output
# ──────────────────────────────────────────────────────────────────────────────
t0 = step_start("Save output")
os.makedirs("outputs", exist_ok=True)
output_path = f"outputs/ltx23_two_stage_{width}x{height}_{seconds}s_seed_{seed}.mp4"
encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=audio_sample_rate,
    output_path=output_path,
)
step_end("Save output", t0)

# ── Summary ───────────────────────────────────────────────────────────────────
total_time = time.time() - script_start
print(f"\n{'═' * 70}")
print(f"  TOTAL: {total_time:.1f}s | Peak VRAM: {global_peak_vram:.2f} GB | Peak RAM: {global_peak_ram:.2f} GB")
print(f"  Output: {output_path}")
print(f"{'═' * 70}")
