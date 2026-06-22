from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from demucs import __version__ as demucs_version
from demucs.pretrained import get_model

warnings.filterwarnings("ignore")

MODEL_NAME = "htdemucs"
EXPECTED_SOURCES = ["drums", "bass", "other", "vocals"]
EXPECTED_SAMPLE_RATE = 44_100
DEFAULT_SEED = 1337

@dataclass
class ExportManifest:
    model_name: str
    file: str
    sources: list[str]
    samplerate: int
    segment: float
    segment_samples: int
    is_bag: bool
    num_models: int
    torch_version: str
    demucs_version: str
    python_version: str
    traced_device: str
    dtype: str

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def models_dir() -> Path:
    path = project_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path

def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, path)
    os.chmod(path, 0o644)

def atomic_save_torchscript(path: Path, traced: torch.jit.ScriptModule) -> None:
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as tmp:
        temp_path = Path(tmp.name)
    try:
        traced.save(str(temp_path))
        os.replace(temp_path, path)
        os.chmod(path, 0o644)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

def load_single_htdemucs():
    model_or_bag = get_model(MODEL_NAME)
    print(f"Loaded model: {MODEL_NAME}")
    print(f"Resolved type: {type(model_or_bag).__name__}")

    if hasattr(model_or_bag, "models"):
        models = list(model_or_bag.models)
        if len(models) != 1:
            raise RuntimeError(
                f"Expected {MODEL_NAME} to resolve to exactly 1 model, got {len(models)}"
            )
        
        model = models[0]
        is_bag = True
        num_models = len(models)
    else:
        model = model_or_bag
        is_bag = False
        num_models = 1

    return model, is_bag, num_models

def validate_model(model) -> tuple[list[str], int, float, int]:
    sources = list(model.sources)
    samplerate = int(model.samplerate)
    segment = float(model.segment)
    segment_samples = int(segment * samplerate)

    if sources != EXPECTED_SOURCES:
        raise RuntimeError(
            f"Unexpected sources order: got {sources}, expected {EXPECTED_SOURCES}"
        )
    
    if samplerate != EXPECTED_SAMPLE_RATE:
        raise RuntimeError(
            f"Unexpected sample rate: got {samplerate}, expected {EXPECTED_SAMPLE_RATE}"
        )
    
    if segment_samples <= 0:
        raise RuntimeError(f"Invalid segment_samples computed: {segment_samples}")

    return sources, samplerate, segment, segment_samples

def prepare_model(model, device: str) -> None:
    model.to(device)
    model.eval()

    for _, buf in model.named_buffers():
        buf.data = buf.data.to(device)

def trace_model(model, segment_samples: int, device: str) -> torch.jit.ScriptModule:
    torch.manual_seed(DEFAULT_SEED)
    
    if device == "cuda":
        torch.set_default_device('cuda')
        
    try:
        example = torch.randn(1, 2, segment_samples, device=device, dtype=torch.float32)

        with torch.no_grad():
            traced = torch.jit.trace(model, example, strict=False)
            traced = torch.jit.freeze(traced)

            output = traced(example)
            expected_shape = (1, 4, 2, segment_samples)
            if tuple(output.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected traced output shape: got {tuple(output.shape)}, expected {expected_shape}"
                )
    finally:
        if device == "cuda":
            torch.set_default_device('cpu')

    return traced

def build_manifest(
    file_name: str,
    sources: list[str],
    samplerate: int,
    segment: float,
    segment_samples: int,
    is_bag: bool,
    num_models: int,
    device: str,
) -> ExportManifest:
    return ExportManifest(
        model_name=MODEL_NAME,
        file=file_name,
        sources=sources,
        samplerate=samplerate,
        segment=segment,
        segment_samples=segment_samples,
        is_bag=is_bag,
        num_models=num_models,
        torch_version=torch.__version__,
        demucs_version=demucs_version,
        python_version=sys.version.split()[0],
        traced_device=device,
        dtype="float32",
    )

def main() -> int:
    parser = argparse.ArgumentParser(description="Export Demucs to TorchScript.")

    parser.add_argument(
        "--device", 
        default="all",
        choices=["all", "cpu", "cuda"],
        help="Target device to export for (default: all available)"
    )

    args = parser.parse_args()
    out_dir = models_dir()
    model, is_bag, num_models = load_single_htdemucs()
    sources, samplerate, segment, segment_samples = validate_model(model)

    print("Model details:")
    print(f"  sources:         {sources}")
    print(f"  samplerate:      {samplerate}")
    print(f"  segment:         {segment}")
    print(f"  segment_samples: {segment_samples}\n")

    export_devices = []
    if args.device in ["all", "cpu"]:
        export_devices.append("cpu")
        
    if args.device in ["all", "cuda"]:
        if torch.cuda.is_available():
            export_devices.append("cuda")
        elif args.device == "cuda":
            print("ERROR: --device cuda requested but no CUDA GPU was detected.", file=sys.stderr)
            sys.exit(1)
        else:
            print("CUDA not detected. Skipping GPU model export.")

    for device in export_devices:
        print(f"--- Exporting for {device.upper()} ---")
        
        suffix = "_cuda" if device == "cuda" else ""
        out_path = out_dir / f"{MODEL_NAME}{suffix}.pt"
        manifest_path = out_dir / f"{MODEL_NAME}{suffix}_manifest.json"

        prepare_model(model, device)

        print(f"Tracing model on {device.upper()} (this takes a moment)...")
        traced = trace_model(model, segment_samples, device)
        atomic_save_torchscript(out_path, traced)

        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"Saved {device.upper()} TorchScript: {out_path} ({size_mb:.0f} MB)")

        manifest = build_manifest(
            file_name=out_path.name,
            sources=sources,
            samplerate=samplerate,
            segment=segment,
            segment_samples=segment_samples,
            is_bag=is_bag,
            num_models=num_models,
            device=device,
        )
        
        atomic_write_json(manifest_path, asdict(manifest))
        print(f"Saved manifest:    {manifest_path}\n")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())