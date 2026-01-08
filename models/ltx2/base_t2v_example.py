import torch
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video


pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=torch.bfloat16)
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
    output_type="np",
    return_dict=False,
)
video = (video * 255).round().astype("uint8")
video = torch.from_numpy(video)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,  # should be 24000
    output_path="video.mp4",
)
