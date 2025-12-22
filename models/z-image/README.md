# Z-IMAGE

Here is a table with the benchmarks for this model. You can also download a [csv file](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/benchmark_results.csv) or you can view it online [here](https://flatgithub.com/asomoza/diffusers-recipes/blob/main/models/z-image/benchmark_results.csv) if you want to apply filters to it.

| Script | Peak RAM (GB) | Peak VRAM (GB) | Total Time (s) | Denoise (s) |
| --- | --- | --- | --- | --- |
| [base_example.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/base_example.py) | 9.76 | 24.87 | 10.03 | 3.00 |
| [model_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/model_cpu_offload.py) | 24.38 | 13.50 | 12.05 | 4.00 |
| [layerwise.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/layerwise.py) | 10.25 | 19.97 | 10.30 | 3.00 |
| [sequential_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/sequential_cpu_offload.py) | 21.02 | 1.86 | 16.26 | 10.00 |
| [pipeline_leaf_stream_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_stream_group_offload.py) | 33.15 | 6.58 | 16.19 | 4.00 |
| [pipeline_leaf_stream_group_offload_record.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_stream_group_offload_record.py) | 33.18 | 11.14 | 15.26 | 3.00 |
| [pipeline_leaf_stream_group_offload_low_mem.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_stream_group_offload_low_mem.py) | 22.26 | 6.58 | 15.38 | 9.00 |
| [pipeline_leaf_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_group_offload.py) | 28.45 | 3.66 | 23.46 | 16.00 |
| [pipeline_block_group_offload_two_blocks.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_block_group_offload_two_blocks.py) | 28.45 | 4.72 | 25.03 | 18.00 |
| [pipeline_leaf_stream_group_offload_disk.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_stream_group_offload_disk.py) | 22.52 | 6.58 | 16.98 | 7.00 |
| [pipeline_leaf_stream_group_offload_disk_record.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/pipeline_leaf_stream_group_offload_disk_record.py) | 22.52 | 6.60 | 17.45 | 7.00 |
| [layerwise_pipeline_leaf_stream_group_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/layerwise_pipeline_leaf_stream_group_offload.py) | 26.54 | 8.50 | 14.12 | 3.00 |
| [layerwise_pipeline_leaf_stream_group_offload_disk.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/layerwise_pipeline_leaf_stream_group_offload_disk.py) | 18.72 | 7.53 | 13.39 | 4.00 |
| [separate_steps.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/separate_steps.py) | 19.61 | 13.50 | 13.56 | 3.00 |
| [gguf_Q8_0.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0.py) | 14.16 | 20.06 | 10.27 | 3.00 |
| [gguf_Q8_0_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0_cpu_offload.py) | 16.32 | 8.53 | 11.80 | 5.00 |
| [gguf_Q8_0_leaf_stream_group_offload_record.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0_leaf_stream_group_offload_record.py) | 25.26 | 8.45 | 14.09 | 4.00 |
| [gguf_Q8_0_leaf_stream_group_offload_disk.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0_leaf_stream_group_offload_disk.py) | 17.78 | 6.75 | 13.97 | 4.00 |
| [gguf_Q8_0_both.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0_both.py) | 24.12 | 16.69 | 13.97 | 3.00 |
| [gguf_Q8_0_both_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q8_0_both_cpu_offload.py) | 23.72 | 8.53 | 15.43 | 4.00 |
| [gguf_Q4_K_M.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q4_K_M.py) | 9.80 | 17.96 | 11.49 | 4.00 |
| [gguf_Q4_K_M_4bit_TE.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q4_K_M_4bit_TE.py) | 9.98 | 12.71 | 11.18 | 4.00 |
| [gguf_Q4_K_M_4bit_TE_cpu_offload.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q4_K_M_4bit_TE_cpu_offload.py) | 9.62 | 6.42 | 12.06 | 5.00 |
| [gguf_Q4_K_M_4bit_TE_leaf_stream_group_offload_record.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/gguf_Q4_K_M_4bit_TE_leaf_stream_group_offload_record.py) | 12.41 | 7.26 | 12.92 | 4.00 |
| [lora.py](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/scripts/lora.py) | 10.20 | 25.05 | 16.96 | 4.00 |