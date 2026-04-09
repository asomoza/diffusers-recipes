import os

import torch
from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from sdnq import SDNQConfig  # noqa: F401
from transformers import Gemma3ForConditionalGeneration


torch_dtype = torch.bfloat16

text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
    "OzzyGT/LTX-2.3-sdnq-dynamic-int4",
    subfolder="text_encoder",
    dtype=torch_dtype,
    device_map="cpu",
)


transformer = LTX2VideoTransformer3DModel.from_pretrained(
    "OzzyGT/LTX-2.3-sdnq-dynamic-int4",
    subfolder="transformer",
    torch_dtype=torch_dtype,
    device_map="cpu",
)

pipe = LTX2Pipeline.from_pretrained(
    "OzzyGT/LTX-2.3", transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
)

pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

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

frame_rate = 24.0
seconds = 5
num_frames = round(seconds * frame_rate) // 8 * 8 + 1

video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=960,
    height=544,
    num_frames=num_frames,
    frame_rate=frame_rate,
    num_inference_steps=30,
    guidance_scale=3.0,
    audio_guidance_scale=7.0,
    guidance_rescale=0.7,
    audio_guidance_rescale=0.7,
    stg_scale=1.0,
    audio_stg_scale=1.0,
    spatio_temporal_guidance_blocks=[28],
    modality_scale=3.0,
    audio_modality_scale=3.0,
    generator=torch.Generator("cuda").manual_seed(42),
    output_type="np",
    return_dict=False,
)
video = (video * 255).round().astype("uint8")
video = torch.from_numpy(video)

if not os.path.exists("./outputs/ltx23"):
    os.makedirs("./outputs/ltx23")

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
    output_path="./outputs/ltx23/t2v_sdnq_4bit_both_cpu_offload.mp4",
)
