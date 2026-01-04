# -*- coding: utf-8 -*-
# Convert MiniFASNet .pth checkpoints to ONNX for inference/benchmarking.

import argparse
import os
from typing import List, Tuple

import torch

from src.anti_spoof_predict import MODEL_MAPPING
from src.utility import get_kernel, parse_model_name


def load_model(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, int, int]:
    """
    Instantiate architecture, load weights, and return model plus expected H, W.
    """
    h_input, w_input, model_type, _ = parse_model_name(os.path.basename(model_path))
    kernel_size = get_kernel(h_input, w_input)
    model_cls = MODEL_MAPPING.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unsupported model type '{model_type}' in {model_path}")
    model = model_cls(conv6_kernel=kernel_size).to(device)

    state_dict = torch.load(model_path, map_location=device)
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, h_input, w_input


def export_to_onnx(model: torch.nn.Module, h: int, w: int, out_path: str, opset: int) -> None:
    dummy = torch.randn(1, 3, h, w, device=next(model.parameters()).device)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def convert(models: List[str], output_dir: str, opset: int, use_gpu: bool) -> None:
    if opset > 10:
        raise ValueError(
            f"Requested opset {opset} is too new for the bundled torch version; "
            "use 9 or 10."
        )
    device = torch.device("cuda:0") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)

    for model_path in models:
        print(f"Converting {model_path} ...")
        model, h, w = load_model(model_path, device)
        base = os.path.splitext(os.path.basename(model_path))[0]
        out_path = os.path.join(output_dir, f"{base}.onnx")
        export_to_onnx(model, h, w, out_path, opset)
        print(f"Saved ONNX to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MiniFASNet .pth checkpoints to ONNX.")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to .pth checkpoints.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./resources/onnx_models",
        help="Destination folder for ONNX files.",
    )
    parser.add_argument("--opset", type=int, default=9, help="ONNX opset version (torch 1.2 supports up to 10).")
    parser.add_argument("--use_gpu", action="store_true", help="Export with CUDA if available.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.models, args.output_dir, args.opset, args.use_gpu)
