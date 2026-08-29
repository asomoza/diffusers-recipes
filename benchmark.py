import atexit
import contextlib
import csv
import ctypes
import gc
import os
import re
import selectors
import shutil
import subprocess
import sys
import time

import psutil
import pynvml


REPORT_TABLES = True
CSV_OUTPUT_PATH = "./benchmark_results.csv"
MD_OUTPUT_PATH = "./benchmark_results.md"
OFFLOAD_TEMP_DIR = "./offload_temp"

# Parse-friendly marker recommended for benchmarked scripts to print:
#   DENOISE_TIME_S=0.123456
_DENOISE_RE = re.compile(r"\bDENOISE_TIME_S\s*=\s*([0-9]+(?:\.[0-9]+)?)\b", re.IGNORECASE)

# Matches the *completed* tqdm bar line, extracting elapsed time from:
#   100%|...| 9/9 [00:03<00:00,  2.51it/s]
# Elapsed can be MM:SS or HH:MM:SS
_TQDM_DONE_RE = re.compile(
    r"100%\|.*?\b(\d+)\s*/\s*\1\b\s*\[\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)\s*<",
    re.IGNORECASE,
)

# Best-effort fallback patterns (if you can't change child scripts)
_FALLBACK_DENOISE_RE_LIST = [
    re.compile(
        r"\bdenois(?:e|ing)\b.*?\btime\b.*?([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|second|seconds)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdenois(?:e|ing)\b.*?([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|secs|second|seconds)\b",
        re.IGNORECASE,
    ),
]


_NVML_INITIALIZED = False
_NVML_DEVICE_HANDLES: list = []


def _parse_clock_to_seconds(s: str) -> float | None:
    # Accept "MM:SS" or "HH:MM:SS"
    try:
        parts = [int(p) for p in s.strip().split(":")]
    except ValueError:
        return None

    if len(parts) == 2:
        mm, ss = parts
        return float(mm * 60 + ss)
    if len(parts) == 3:
        hh, mm, ss = parts
        return float(hh * 3600 + mm * 60 + ss)
    return None


def _nvml_init_if_needed() -> bool:
    global _NVML_INITIALIZED, _NVML_DEVICE_HANDLES

    if _NVML_INITIALIZED:
        return True
    try:
        pynvml.nvmlInit()
        _NVML_DEVICE_HANDLES = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        _NVML_INITIALIZED = True

        def _shutdown() -> None:
            with contextlib.suppress(pynvml.NVMLError):
                pynvml.nvmlShutdown()

        atexit.register(_shutdown)
    except pynvml.NVMLError:
        return False
    else:
        return True


def _nvml_pid_vram_bytes_on_handle(handle, pid: int) -> int:
    used = 0

    def _get_procs(fn_name: str):
        fn = getattr(pynvml, fn_name, None)

        if fn is None:
            return []

        try:
            return fn(handle)
        except pynvml.NVMLError:
            return []

    procs = []
    procs.extend(_get_procs("nvmlDeviceGetComputeRunningProcesses_v2"))
    procs.extend(_get_procs("nvmlDeviceGetGraphicsRunningProcesses_v2"))

    for p in procs:
        try:
            if int(p.pid) == int(pid):
                m = int(getattr(p, "usedGpuMemory", 0) or 0)
                if m > 0:
                    used += m
        except (AttributeError, TypeError, ValueError):
            continue

    return used


def _get_process_vram_used_gb(pid: int) -> float:
    if not _nvml_init_if_needed():
        return 0.0

    total_bytes = 0

    for h in _NVML_DEVICE_HANDLES:
        total_bytes += _nvml_pid_vram_bytes_on_handle(h, pid)

    return total_bytes / (1024**3)


def _malloc_trim() -> None:
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def _cleanup_between_runs() -> None:
    gc.collect()
    _malloc_trim()
    time.sleep(0.05)


def _extract_denoise_time_s_from_text(text: str) -> float | None:
    # Prefer explicit marker if it exists; if multiple are present, take the last one.
    last_marker: float | None = None
    for m in _DENOISE_RE.finditer(text):
        try:
            last_marker = float(m.group(1))
        except ValueError:
            continue
    if last_marker is not None:
        return last_marker

    # Next, parse the last completed tqdm bar (usually the inference steps bar).
    last_tqdm: float | None = None
    for m in _TQDM_DONE_RE.finditer(text):
        t = _parse_clock_to_seconds(m.group(2))
        if t is not None:
            last_tqdm = t
    if last_tqdm is not None:
        return last_tqdm

    # Finally, try fuzzy “denoise time: X s” patterns.
    for rx in _FALLBACK_DENOISE_RE_LIST:
        last_fuzzy: float | None = None
        for m2 in rx.finditer(text):
            try:
                last_fuzzy = float(m2.group(1))
            except ValueError:
                continue
        if last_fuzzy is not None:
            return last_fuzzy

    return None


def measure_memory_usage(script_path: str):
    _cleanup_between_runs()

    start_time = time.time()

    p = subprocess.Popen(  # noqa: S603 - script_path comes from the hardcoded list at the bottom of this file
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    try:
        proc = psutil.Process(p.pid)
    except psutil.Error:
        proc = None

    peak_rss_gb = 0.0
    peak_vram_gb = 0.0
    denoise_s: float | None = None

    sel = selectors.DefaultSelector()
    stdout_fd = None
    if p.stdout is not None:
        stdout_fd = p.stdout.fileno()
        sel.register(stdout_fd, selectors.EVENT_READ)

    text_buf = ""

    while True:
        ret = p.poll()

        if proc is not None:
            try:
                peak_rss_gb = max(peak_rss_gb, proc.memory_info().rss / (1024**3))
            except psutil.Error:
                proc = None

        peak_vram_gb = max(peak_vram_gb, _get_process_vram_used_gb(p.pid))

        if stdout_fd is not None:
            events = sel.select(timeout=0)
            for key, _ in events:
                try:
                    chunk = os.read(key.fd, 4096)
                except OSError:
                    chunk = b""

                if chunk:
                    with contextlib.suppress(OSError, ValueError):
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.buffer.flush()

                    decoded = chunk.decode("utf-8", errors="replace")
                    decoded = decoded.replace("\r", "\n")
                    text_buf += decoded

                    while "\n" in text_buf:
                        line, text_buf = text_buf.split("\n", 1)
                        maybe = _extract_denoise_time_s_from_text(line)
                        if maybe is not None:
                            denoise_s = maybe
                else:
                    with contextlib.suppress(KeyError, ValueError):
                        sel.unregister(key.fd)
                    stdout_fd = None

        if ret is not None:
            if stdout_fd is not None:
                try:
                    while True:
                        chunk = os.read(stdout_fd, 4096)
                        if not chunk:
                            break
                        with contextlib.suppress(OSError, ValueError):
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                        decoded = chunk.decode("utf-8", errors="replace").replace("\r", "\n")
                        maybe = _extract_denoise_time_s_from_text(decoded)
                        if maybe is not None:
                            denoise_s = maybe
                except (OSError, ValueError):
                    pass
            break

        time.sleep(0.05)

    end_time = time.time()
    _cleanup_between_runs()

    return (peak_rss_gb, peak_vram_gb, end_time - start_time, denoise_s, int(p.returncode or 0))


def _format_markdown_table(results: list[dict]) -> str:
    headers = ["Script", "Peak RAM (GB)", "Peak VRAM (GB)", "Time (s)", "Denoise (s)"]
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in results:
        denoise_str = "" if r.get("denoise_s") is None else f"{r['denoise_s']:.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["script"]),
                    f"{r['peak_ram_gb']:.2f}",
                    f"{r['peak_vram_gb']:.2f}",
                    f"{r['duration_s']:.2f}",
                    denoise_str,
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def _write_markdown(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(_format_markdown_table(results))


def _write_csv(results: list[dict], path: str) -> None:
    fieldnames = ["script", "peak_ram_gb", "peak_vram_gb", "duration_s", "denoise_s"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})


if __name__ == "__main__":
    script_paths = [
        "./models/z-image/scripts/base_example.py",
        "./models/z-image/scripts/model_cpu_offload.py",
        "./models/z-image/scripts/layerwise.py",
        "./models/z-image/scripts/sequential_cpu_offload.py",
        "./models/z-image/scripts/pipeline_leaf_stream_group_offload.py",
        "./models/z-image/scripts/pipeline_leaf_stream_group_offload_record.py",
        "./models/z-image/scripts/pipeline_leaf_stream_group_offload_low_mem.py",
        "./models/z-image/scripts/pipeline_leaf_group_offload.py",
        "./models/z-image/scripts/pipeline_block_group_offload_two_blocks.py",
        "./models/z-image/scripts/pipeline_leaf_stream_group_offload_disk.py",
        "./models/z-image/scripts/pipeline_leaf_stream_group_offload_disk_record.py",
        "./models/z-image/scripts/layerwise_pipeline_leaf_stream_group_offload.py",
        "./models/z-image/scripts/layerwise_pipeline_leaf_stream_group_offload_disk.py",
        "./models/z-image/scripts/separate_steps.py",
        "./models/z-image/scripts/gguf_Q8_0.py",
        "./models/z-image/scripts/gguf_Q8_0_cpu_offload.py",
        "./models/z-image/scripts/gguf_Q8_0_leaf_stream_group_offload_record.py",
        "./models/z-image/scripts/gguf_Q8_0_leaf_stream_group_offload_disk.py",
        "./models/z-image/scripts/gguf_Q8_0_both.py",
        "./models/z-image/scripts/gguf_Q8_0_both_cpu_offload.py",
        "./models/z-image/scripts/gguf_Q4_K_M.py",
        "./models/z-image/scripts/gguf_Q4_K_M_4bit_TE.py",
        "./models/z-image/scripts/gguf_Q4_K_M_4bit_TE_cpu_offload.py",
        "./models/z-image/scripts/gguf_Q4_K_M_4bit_TE_leaf_stream_group_offload_record.py",
    ]

    results: list[dict] = []

    for script in script_paths:
        peak_ram_gb, peak_vram_gb, duration, denoise_s, exit_code = measure_memory_usage(script)

        print(f"\nScript: {script}")
        print(f"Peak RAM Usage: {peak_ram_gb:.2f} GB")
        print(f"Peak VRAM Usage: {peak_vram_gb:.2f} GB")
        print(f"Execution Time: {duration:.2f} seconds")
        print(f"Denoise Time: {'' if denoise_s is None else f'{denoise_s:.2f} seconds'}")
        print("-" * 40)

        results.append(
            {
                "script": script,
                "peak_ram_gb": float(peak_ram_gb),
                "peak_vram_gb": float(peak_vram_gb),
                "duration_s": float(duration),
                "denoise_s": (None if denoise_s is None else float(denoise_s)),
            }
        )

        shutil.rmtree(OFFLOAD_TEMP_DIR, ignore_errors=True)
        os.makedirs(OFFLOAD_TEMP_DIR, exist_ok=True)

    if REPORT_TABLES:
        _write_markdown(results, MD_OUTPUT_PATH)
        _write_csv(results, CSV_OUTPUT_PATH)
        print(f"Wrote Markdown: {MD_OUTPUT_PATH}")
        print(f"Wrote CSV: {CSV_OUTPUT_PATH}")
