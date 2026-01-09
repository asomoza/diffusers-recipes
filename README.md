# DIFFUSERS RECIPES

In this repository, you will find multiple files with examples for the most popular models supported by diffusers.

## HOW TO USE THIS REPOSITORY

These recipes are designed so you can just copy and paste them into your environment.

### CURRENT MODELS

* [Z-Image](https://github.com/asomoza/diffusers-recipes/blob/main/models/z-image/README.md)

### USING THIS REPOSITORY

If you want, you can also run the scripts directly from this repository.

First, clone the project:

```bash
git clone https://github.com/asomoza/diffusers-recipes.git
```

Then you will need to install `uv` by following the [official instructions](https://docs.astral.sh/uv/) for your system.

After installing `uv`, you can install the basic packages with this command:

```bash
uv sync
```

This will use Python 3.12 and install the recommended packages to run Mellon. It may also install `torch >2.9` with `CUDA >13.0` support, depending on your OS.

Note: This command will also install `diffusers` from main.

## RUNNING THE SCRIPTS

From the console, you can run each script with the following command:

```bash
uv run <script>
```

For example:

```bash
uv run models/z-image/scripts/base_example.py
```

For scripts that use quantization, you will need to install the quantization extras with this command:

```bash
uv sync --extra quantization
```

For scripts that use a different attention backend, you need to install the attention extras:

```bash
uv sync --extra attentions
```

For scripts that use video, you need to install the video extras:

```bash
uv sync --extra video
```

## BENCHMARKS

The benchmarks were run on the following systems:

```
Test Bench #1

AMD Ryzen 7 9800X3D 8-Core Processor
128GB of RAM
RTX 5090 32GB of RAM (undervolted to 480 W)
PCIe Gen5 NVMe 14,900 MB/s
```
