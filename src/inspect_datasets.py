"""
Hardware Detection and Dataset Inspection Module for EMP-26.
Checks:
1. Hardware / GPU specifications (CUDA, GPU name, VRAM, PyTorch version, CUDA version)
2. Hugging Face dataset schemas and connectivity
"""

import sys
import os
from pathlib import Path
import yaml
import torch

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def detect_hardware():
    print("\n" + "=" * 60)
    print("HARDWARE & ACCELERATION DETECTION")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None (CPU Execution)"
    vram = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if cuda_available else 0.0
    pytorch_version = torch.__version__
    cuda_version = torch.version.cuda if cuda_available else "N/A"

    print(f"CUDA available:     {cuda_available}")
    print(f"GPU Device Count:   {device_count}")
    print(f"GPU Name:           {gpu_name}")
    print(f"VRAM:               {vram:.2f} GB")
    print(f"PyTorch Version:    {pytorch_version}")
    print(f"CUDA Version:       {cuda_version}")

    device = "cuda" if cuda_available else "cpu"
    print(f"Active Compute:     {device.upper()}")
    print("=" * 60 + "\n")
    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_gb": vram,
        "pytorch_version": pytorch_version,
        "cuda_version": cuda_version,
        "device": device,
    }


def inspect_datasets():
    print("=" * 60)
    print("HUGGING FACE DATASET SCHEMA INSPECTION")
    print("=" * 60)
    try:
        from datasets import load_dataset
    except ImportError:
        print("[Error] Hugging Face 'datasets' library is not installed.")
        return

    datasets_to_check = [
        ("Omarrran/StackPulse_778K_QnA_Code_dataset", "high_quality"),
        ("stindardlogic/coding-interview-sft-100k", None),
        ("Vineeshsuiii/Software_Engineering_interview_datasets", None),
        ("nefro313/hr_questions_common_and_answers_", None),
    ]

    for name, config in datasets_to_check:
        print(f"\n[Inspecting] {name} (Config: {config or 'default'})...")
        try:
            if config:
                ds = load_dataset(name, config, split="train", streaming=True)
            else:
                ds = load_dataset(name, split="train", streaming=True)
            sample = next(iter(ds))
            print(f"  ✓ Connected! Columns: {list(sample.keys())}")
            for k in list(sample.keys())[:4]:
                preview = str(sample[k])[:80].replace("\n", " ")
                print(f"    - {k}: {preview}...")
        except Exception as e:
            print(f"  ✗ Connection/Inspection warning: {e}")

    print("\n" + "=" * 60)
    print("Inspection complete.")
    print("=" * 60)


if __name__ == "__main__":
    detect_hardware()
    inspect_datasets()
