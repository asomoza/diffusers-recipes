import os

import torch
from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model
from transformers import Gemma3ForConditionalGeneration


torch_dtype = torch.bfloat16

text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
    "Disty0/LTX-2-SDNQ-4bit-dynamic",
    subfolder="text_encoder",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)


transformer = LTX2VideoTransformer3DModel.from_pretrained(
    "Disty0/LTX-2-SDNQ-4bit-dynamic",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2", transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16
)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

prompt = """Cinematic video with professional lighting, shot with a 35mm lens and shallow depth of field. The scene opens on an extreme close-up of a real car license plate reading “DIFFUSERS,” captured with macro focus, rock music kicks in and builds intensity. The setting is an outdoor wet mountain road, surrounded by dark silhouettes of hills, trees, and winding asphalt. The car is a high-performance modern sports car with large performance tires gripping the rain-soaked road. A deep combustion engine growl builds as the car suddenly accelerates forward."""
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

frame_rate = 24.0
video, audio = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=121,
    frame_rate=frame_rate,
    num_inference_steps=40,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
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
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
    output_path="./outputs/ltx2/t2v_sdnq-4bit-both.mp4",
)
