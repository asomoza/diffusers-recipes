# Z-IMAGE

Here is a table with the benchmarks for this model. You can also download a [csv file](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/benchmark_results.csv) or you can view it online [here](https://flatgithub.com/asomoza/diffusers-recipes/blob/main/models/z-image/benchmark_results.csv) if you want to apply filters to it.

| Script | Peak RAM (GB) | Peak VRAM (GB) | Total Time (s) | Denoise (s) |
| --- | --- | --- | --- | --- |
| [base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/base_example.py) | 10.03 | 24.87 | 12.32 | 3.00 |
| [model_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/model_cpu_offload.py) | 24.39 | 13.50 | 12.04 | 4.00 |
| [layerwise.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/layerwise.py) | 9.85 | 19.96 | 10.13 | 3.00 |
| [sequential_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/sequential_cpu_offload.py) | 21.07 | 2.37 | 16.24 | 10.00 |
| [pipeline_leaf_stream_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_stream_group_offload.py) | 33.17 | 6.58 | 16.01 | 4.00 |
| [pipeline_leaf_stream_group_offload_record.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_stream_group_offload_record.py) | 33.26 | 11.10 | 15.32 | 3.00 |
| [pipeline_leaf_stream_group_offload_low_mem.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_stream_group_offload_low_mem.py) | 22.27 | 6.58 | 15.23 | 9.00 |
| [pipeline_leaf_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_group_offload.py) | 26.30 | 3.66 | 23.52 | 16.00 |
| [pipeline_block_group_offload_two_blocks.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_block_group_offload_two_blocks.py) | 28.35 | 4.72 | 25.15 | 18.00 |
| [pipeline_leaf_stream_group_offload_disk.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_stream_group_offload_disk.py) | 21.08 | 6.58 | 17.03 | 7.00 |
| [pipeline_leaf_stream_group_offload_disk_record.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_stream_group_offload_disk_record.py) | 18.36 | 6.60 | 13.55 | 7.00 |
| [layerwise_pipeline_leaf_stream_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/layerwise_pipeline_leaf_stream_group_offload.py) | 26.77 | 8.52 | 14.13 | 3.00 |
| [layerwise_pipeline_leaf_stream_group_offload_disk.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/layerwise_pipeline_leaf_stream_group_offload_disk.py) | 18.27 | 6.60 | 14.20 | 7.00 |
| [separate_steps.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/separate_steps.py) | 19.93 | 13.50 | 13.95 | 3.00 |
| [gguf_Q8_0.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0.py) | 13.33 | 20.06 | 10.17 | 3.00 |
| [gguf_Q8_0_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0_cpu_offload.py) | 16.35 | 8.53 | 11.78 | 5.00 |
| [gguf_Q8_0_leaf_stream_group_offload_record.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0_leaf_stream_group_offload_record.py) | 25.27 | 8.46 | 14.01 | 4.00 |
| [gguf_Q8_0_leaf_stream_group_offload_disk.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0_leaf_stream_group_offload_disk.py) | 13.99 | 2.63 | 5.67 | 0.00 |
| [gguf_Q8_0_both.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0_both.py) | 23.83 | 16.69 | 13.65 | 3.00 |
| [gguf_Q8_0_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0_both_cpu_offload.py) | 24.38 | 8.53 | 15.03 | 4.00 |
| [gguf_Q4_K_M.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q4_K_M.py) | 9.26 | 17.96 | 10.67 | 4.00 |
| [gguf_Q4_K_M_4bit_TE.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q4_K_M_4bit_TE.py) | 9.56 | 12.71 | 10.77 | 4.00 |
| [gguf_Q4_K_M_4bit_TE_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q4_K_M_4bit_TE_cpu_offload.py) | 9.60 | 6.42 | 11.94 | 5.00 |
| [gguf_Q4_K_M_4bit_TE_leaf_stream_group_offload_record.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q4_K_M_4bit_TE_leaf_stream_group_offload_record.py) | 12.42 | 7.26 | 12.91 | 4.00 |
