import gc
import os

import torch
from diffusers import AutoencoderKLLTX2Video, LTX2LatentUpsamplePipeline, LTX2Pipeline, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import Gemma3ForConditionalGeneration


torch_dtype = torch.bfloat16
onload_device = torch.device("cuda")
offload_device = torch.device("cpu")
model_path = "Lightricks/LTX-2"
width = 768
height = 512
num_frames = 481
frame_rate = 24.0
seed = int.from_bytes(os.urandom(8), "big")
generator = torch.Generator(onload_device).manual_seed(seed)

prompt = """"An action packed, cinematic shot of a monster truck driving fast towards the camera, the truck passes the cameras it pans left to follow the trucks reckless drive. dust and motion blur is around the truck, hand held feel to the camera as it tries to track its ride into the distance. the truck then drifts and turns around, then drives back towards the camera until seen in extreme close up."""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

text_encoder = Gemma3ForConditionalGeneration.from_pretrained("OzzyGT/LTX-2-bnb-8bit-text-encoder", dtype=torch_dtype)

# encode prompt
embeds_pipe = LTX2Pipeline.from_pretrained(
    model_path,
    text_encoder=text_encoder,
    transformer=None,
    vae=None,
    audio_vae=None,
    vocoder=None,
    scheduler=None,
    connectors=None,
    torch_dtype=torch_dtype,
)
embeds_pipe.enable_sequential_cpu_offload()

with torch.inference_mode():
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = embeds_pipe.encode_prompt(prompt, negative_prompt)

# Break any ties to the text-encoder execution buffers and park embeddings on CPU.
prompt_embeds = prompt_embeds.detach().to(offload_device, copy=True)
prompt_attention_mask = prompt_attention_mask.detach().to(offload_device, copy=True)
negative_prompt_embeds = negative_prompt_embeds.detach().to(offload_device, copy=True)
negative_prompt_attention_mask = negative_prompt_attention_mask.detach().to(offload_device, copy=True)

del text_encoder
del embeds_pipe
gc.collect()
torch.cuda.empty_cache()

transformer = LTX2VideoTransformer3DModel.from_pretrained(
    "OzzyGT/LTX-2-bnb-4bit-transformer",
    torch_dtype=torch_dtype,
    device_map="cpu",
)
first_stage_pipe = LTX2Pipeline.from_pretrained(
    model_path, transformer=transformer, text_encoder=None, vocoder=None, torch_dtype=torch_dtype
)
first_stage_pipe.enable_group_offload(
    onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", low_cpu_mem_usage=True
)

prompt_embeds = prompt_embeds.to(onload_device, non_blocking=True)
prompt_attention_mask = prompt_attention_mask.to(onload_device, non_blocking=True)
negative_prompt_embeds = negative_prompt_embeds.to(onload_device, non_blocking=True)
negative_prompt_attention_mask = negative_prompt_attention_mask.to(onload_device, non_blocking=True)

video_latent, audio_latent = first_stage_pipe(
    prompt_embeds=prompt_embeds,
    prompt_attention_mask=prompt_attention_mask,
    negative_prompt_embeds=negative_prompt_embeds,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
    width=width,
    height=height,
    num_frames=num_frames,
    frame_rate=frame_rate,
    num_inference_steps=40,
    guidance_scale=4.0,
    generator=generator,
    output_type="latent",
    return_dict=False,
)

video_latent = video_latent.detach().to(offload_device, copy=True)
audio_latent = audio_latent.detach().to(offload_device, copy=True)
prompt_embeds = prompt_embeds.detach().to(offload_device, copy=True)
prompt_attention_mask = prompt_attention_mask.detach().to(offload_device, copy=True)
negative_prompt_embeds = negative_prompt_embeds.detach().to(offload_device, copy=True)
negative_prompt_attention_mask = negative_prompt_attention_mask.detach().to(offload_device, copy=True)

del first_stage_pipe
del transformer
gc.collect()
torch.cuda.empty_cache()

latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
    "rootonchair/LTX-2-19b-distilled",
    subfolder="latent_upsampler",
    torch_dtype=torch_dtype,
)

vae = AutoencoderKLLTX2Video.from_pretrained(
    model_path,
    subfolder="vae",
    torch_dtype=torch_dtype,
)

upsample_pipe = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler)
upsample_pipe.enable_model_cpu_offload(device=onload_device)
upscaled_video_latent = upsample_pipe(
    latents=video_latent,
    output_type="latent",
    return_dict=False,
)[0]

upscaled_video_latent = upscaled_video_latent.detach().to(offload_device, copy=True)
del video_latent
del upsample_pipe
del latent_upsampler
del vae
gc.collect()
torch.cuda.empty_cache()

transformer = LTX2VideoTransformer3DModel.from_pretrained(
    "OzzyGT/LTX-2-bnb-4bit-transformer-distilled",
    torch_dtype=torch_dtype,
    device_map="cpu",
)
second_stage_pipe = LTX2Pipeline.from_pretrained(
    model_path, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype
)

stage_2_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    second_stage_pipe.scheduler.config,
    use_dynamic_shifting=False,
    shift_terminal=None,
)
second_stage_pipe.scheduler = stage_2_scheduler

second_stage_pipe.enable_group_offload(
    onload_device=onload_device, offload_device=offload_device, offload_type="leaf_level", low_cpu_mem_usage=True
)

upscaled_video_latent = upscaled_video_latent.to(onload_device, non_blocking=True)
audio_latent = audio_latent.to(onload_device, non_blocking=True)
prompt_embeds = prompt_embeds.to(onload_device, non_blocking=True)
prompt_attention_mask = prompt_attention_mask.to(onload_device, non_blocking=True)
negative_prompt_embeds = negative_prompt_embeds.to(onload_device, non_blocking=True)
negative_prompt_attention_mask = negative_prompt_attention_mask.to(onload_device, non_blocking=True)

video_latent_stage2, audio_latent_stage2 = second_stage_pipe(
    latents=upscaled_video_latent,
    audio_latents=audio_latent,
    prompt_embeds=prompt_embeds,
    prompt_attention_mask=prompt_attention_mask,
    negative_prompt_embeds=negative_prompt_embeds,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
    width=width * 2,
    height=height * 2,
    num_frames=num_frames,
    num_inference_steps=3,
    noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
    sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
    generator=generator,
    guidance_scale=1.0,
    output_type="latent",
    return_dict=False,
)

video_latent_stage2 = video_latent_stage2.detach().to(offload_device, copy=True)
audio_latent_stage2 = audio_latent_stage2.detach().to(offload_device, copy=True)
del second_stage_pipe
del transformer
del upscaled_video_latent
del audio_latent
del prompt_embeds
del prompt_attention_mask
del negative_prompt_embeds
del negative_prompt_attention_mask
gc.collect()
torch.cuda.empty_cache()

decode_pipe = LTX2Pipeline.from_pretrained(
    model_path,
    text_encoder=None,
    transformer=None,
    scheduler=None,
    connectors=None,
    torch_dtype=torch_dtype,
)
decode_pipe.to(onload_device)

decode_pipe.vae.enable_tiling(
    tile_sample_min_height=256,
    tile_sample_min_width=256,
    tile_sample_min_num_frames=16,
    tile_sample_stride_height=192,
    tile_sample_stride_width=192,
    tile_sample_stride_num_frames=8,
)
decode_pipe.vae.use_framewise_encoding = True
decode_pipe.vae.use_framewise_decoding = True
decode_pipe.enable_model_cpu_offload()

with torch.inference_mode():
    decode_video_latents = video_latent_stage2.to(onload_device, dtype=decode_pipe.vae.dtype, non_blocking=True)
    decode_audio_latents = audio_latent_stage2.to(onload_device, dtype=decode_pipe.audio_vae.dtype, non_blocking=True)

    video = decode_pipe.vae.decode(decode_video_latents, None, return_dict=False)[0]
    video = decode_pipe.video_processor.postprocess_video(video, output_type="np")

    generated_mel_spectrograms = decode_pipe.audio_vae.decode(decode_audio_latents, return_dict=False)[0]
    audio = decode_pipe.vocoder(generated_mel_spectrograms)

video = (video * 255).round().astype("uint8")
video = torch.from_numpy(video)

if not os.path.exists("./outputs/ltx2"):
    os.makedirs("./outputs/ltx2")

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=decode_pipe.vocoder.config.output_sampling_rate,
    output_path="./outputs/ltx2/t2v_2_stages_bnb_4bit_quality_steps.mp4",
)
