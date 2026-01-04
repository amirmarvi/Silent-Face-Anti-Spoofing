# -*- coding: utf-8 -*-
# Benchmark ONNX depth models (e.g., MiDaS) on a single image.
# Applies basic resize + normalize preprocessing, times inference with warmup/repeat,
# and optionally writes a CSV report.

import argparse
import csv
import os
import time
from typing import List, Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort


def load_image(path: str, color_order: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image at {path}")
    if color_order.lower() == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def preprocess(
    image: np.ndarray,
    out_h: int,
    out_w: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = x.transpose(2, 0, 1)[np.newaxis, ...]
    return x


def run_session(session: ort.InferenceSession, input_blob: np.ndarray, repeat: int, warmup: int) -> float:
    ort_inputs = {session.get_inputs()[0].name: input_blob}
    for _ in range(max(warmup, 0)):
        session.run(None, ort_inputs)
    start = time.perf_counter()
    for _ in range(max(repeat, 1)):
        _ = session.run(None, ort_inputs)
    elapsed = time.perf_counter() - start
    return (elapsed / max(repeat, 1)) * 1000.0


def summarize_output(output: np.ndarray) -> str:
    arr = output.astype(np.float32)
    return f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"


def benchmark_models(
    image_path: str,
    model_paths: Sequence[str],
    input_h: int,
    input_w: int,
    mean: Sequence[float],
    std: Sequence[float],
    color_order: str,
    repeat: int,
    warmup: int,
    use_gpu: bool,
    save_report: Optional[str],
) -> None:
    image = load_image(image_path, color_order)
    input_blob = preprocess(image, input_h, input_w, mean, std)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

    results = []
    for path in model_paths:
        session = ort.InferenceSession(path, providers=providers)
        avg_ms = run_session(session, input_blob, repeat, warmup)
        output = session.run(None, {session.get_inputs()[0].name: input_blob})[0]
        stats = summarize_output(output)
        results.append((path, avg_ms, stats))

    print(f"Processed image: {image_path}")
    for path, avg_ms, stats in results:
        print(f"- {os.path.basename(path)} -> avg inference {avg_ms:.2f} ms over {repeat} run(s) (warmup {warmup}); {stats}")

    if save_report:
        os.makedirs(os.path.dirname(save_report) or ".", exist_ok=True)
        with open(save_report, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["image", "model", "avg_ms", "repeat", "warmup", "output_stats"])
            for path, avg_ms, stats in results:
                writer.writerow([image_path, os.path.basename(path), f"{avg_ms:.4f}", repeat, warmup, stats])
        print(f"Saved report to {save_report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX depth models (e.g., MiDaS) on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image.")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to ONNX models to benchmark.")
    parser.add_argument("--height", type=int, default=512, help="Model input height.")
    parser.add_argument("--width", type=int, default=512, help="Model input width.")
    parser.add_argument(
        "--mean",
        nargs=3,
        type=float,
        default=[0.5, 0.5, 0.5],
        help="Normalization mean (per channel).",
    )
    parser.add_argument(
        "--std",
        nargs=3,
        type=float,
        default=[0.5, 0.5, 0.5],
        help="Normalization std (per channel).",
    )
    parser.add_argument(
        "--color_order",
        type=str,
        choices=["rgb", "bgr"],
        default="rgb",
        help="Color order expected by the model.",
    )
    parser.add_argument("--repeat", type=int, default=10, help="Timed runs per model.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per model before timing.")
    parser.add_argument("--use_gpu", action="store_true", help="Use CUDAExecutionProvider if available.")
    parser.add_argument("--save_report", type=str, help="Optional CSV path to store benchmark results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark_models(
        image_path=args.image,
        model_paths=args.models,
        input_h=args.height,
        input_w=args.width,
        mean=args.mean,
        std=args.std,
        color_order=args.color_order,
        repeat=args.repeat,
        warmup=args.warmup,
        use_gpu=args.use_gpu,
        save_report=args.save_report,
    )
