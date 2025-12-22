# QWEN IMAGE LAYERED

Here is a table with the benchmarks for this model. You can also download a [csv file](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen-image-layered/benchmark_results.csv) or you can view it online [here](https://flatgithub.com/asomoza/diffusers-recipes/blob/main/models/qwen-image-layered/benchmark_results.csv) if you want to apply filters to it.

| Script | Peak RAM (GB) | Peak VRAM (GB) | Total Time (s) | Denoise (s) |
| --- | --- | --- | --- | --- |
| [base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen-image-layered/scripts/base_example.py) | OOM | OOM | OOM | OOM |
| [bnb-4bit-both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen-image-layered/scripts/bnb-4bit-both_cpu_offload.py) | 23.54 | 18.68 | 158.18 | 140.00 |
| [torchao-8bit-float8-transformer_only_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen-image-layered/scripts/torchao-8bit-float8-transformer_only_cpu_offload.py) | 37.19 | 23.21 | 135.13 | 122.00 |