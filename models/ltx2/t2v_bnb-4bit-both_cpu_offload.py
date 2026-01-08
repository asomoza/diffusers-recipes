import os

import torch
from diffusers import LTX2Pipeline, LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2.export_utils import encode_video
from transformers import Gemma3ForConditionalGeneration


torch_dtype = torch.bfloat16

text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
    "OzzyGT/LTX-2-bnb-4bit-text-encoder",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

transformer = LTX2VideoTransformer3DModel.from_pretrained(
    "OzzyGT/LTX-2-bnb-4bit-transformer",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)


pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2", transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16
)
pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

prompt = "a fantasy style video of a majestic dragon flying over mountains during sunset"
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
    output_path="./outputs/ltx2/t2v_bnb-4bit-both.mp4",
)
