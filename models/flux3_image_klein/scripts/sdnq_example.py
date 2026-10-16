import os

import torch
from diffusers import ModularPipeline
from sdnq import SDNQConfig  # noqa: F401
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model


device = "cuda"
dtype = torch.bfloat16
SDNQ_BITS = 8  # 4 or 8
repo_id = f"OzzyGT/Flux_3_Image_Klein_sdnq_dynamic_{SDNQ_BITS}bit"
output_dir = "./outputs/flux3_image_klein"
seed = 7
prompt = r"""A cinematic film still of the inside of a small neighborhood pet shop, shot on an ARRI Alexa Mini LF with a Cooke S4 32mm lens at T2.8, 180-degree shutter, camera at eye level on a dolly, wide 16:9 frame. Late-afternoon sunlight comes in low through the front window on the right in hard shafts with dust in the air, and the rest of the shop is lit by practical tungsten ceiling lamps, leaving the corners in shadow. Shallow depth of field, the back wall slightly out of focus, visible film grain, subtle halation around the window highlights. A dark green painted wooden sign <sign_1> with chipped paint hangs high on the back wall under a small brass picture lamp and reads "WHISKERS & WAGS". On a shelf at the left, an adult tabby cat <cat_1> lies next to bags of pet food. In the left foreground, an adult golden retriever <dog_1>, the biggest animal in the shop, sits in a large wicker basket and looks toward the window. In the center, a chalkboard easel <chalkboard_1> reads "ADOPT A FRIEND TODAY" in white chalk. On the right, a scarlet macaw <parrot_1> with a red head and body and blue and yellow wings perches on a wooden stand above a lit aquarium <aquarium_1> with small orange and blue fish. On the floor in a patch of sunlight, an adult white dwarf rabbit <rabbit_1>, smaller than the cat, nibbles a carrot on scuffed checkered tiles. The shelves hold pet toys, leashes and food bags <background_1>. [{"id": "background_1", "bbox": [0, 0, 1000, 1000], "desc": "A pet shop interior with scuffed wooden shelves of pet toys, leashes and food bags, a checkered tile floor, low sunlight through the front window on the right and tungsten ceiling lamps, shadowed corners."}, {"id": "sign_1", "bbox": [40, 300, 150, 700], "desc": "A dark green painted wooden sign with chipped edges, lit from above by a small brass picture lamp, with the text \"WHISKERS & WAGS\" in cream serif letters."}, {"id": "cat_1", "bbox": [290, 50, 440, 230], "desc": "An adult tabby cat lying on a wooden shelf next to bags of pet food."}, {"id": "dog_1", "bbox": [480, 80, 960, 340], "desc": "An adult golden retriever sitting in a large round wicker basket, looking toward the window, the biggest animal in the scene."}, {"id": "chalkboard_1", "bbox": [520, 420, 930, 560], "desc": "A wooden chalkboard easel with \"ADOPT A FRIEND TODAY\" handwritten in white chalk."}, {"id": "parrot_1", "bbox": [180, 760, 440, 870], "desc": "A scarlet macaw with a red head and body and blue and yellow wings, perched on a wooden stand."}, {"id": "aquarium_1", "bbox": [450, 650, 700, 900], "desc": "A lit glass aquarium with green plants and small orange and blue fish."}, {"id": "rabbit_1", "bbox": [850, 620, 960, 700], "desc": "An adult white dwarf rabbit, smaller than the cat, nibbling an orange carrot on the floor in a patch of sunlight."}]"""

if not seed:
    seed = torch.randint(0, 2**32, (1,)).item()
generator = torch.Generator(device="cpu").manual_seed(seed)

# the prompt is already a full caption, so build the pipeline without the prompt upsampler step
blocks = ModularPipeline.from_pretrained(repo_id).blocks
blocks.sub_blocks.pop("prompt_upsampler")
pipe = blocks.init_pipeline(repo_id)
pipe.load_components(dtype=dtype)

if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

pipe.to(device)

image = pipe(
    prompt=prompt,
    height=1152,
    width=2048,
    generator=generator,
    output="images",
)[0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image.save(os.path.join(output_dir, f"sdnq_example_{SDNQ_BITS}bit_{seed}.png"))
