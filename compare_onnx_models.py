# -*- coding: utf-8 -*-
# Simple ONNX runtime comparison for anti-spoofing models.
# Loads one or more ONNX models, runs them on a single image, and reports
# per-model probabilities plus the averaged verdict.

import argparse
import glob
import os
import time
import warnings
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from src.anti_spoof_predict import Detection
from src.generate_patches import CropImage
from src.utility import parse_model_name


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def check_image_ratio(image: np.ndarray) -> None:
    """Warn if the image ratio diverges from the expected 3:4."""
    height, width, _ = image.shape
    ratio = width / height
    if abs(ratio - 0.75) > 1e-3:
        warnings.warn(
            f"Image ratio {ratio:.3f} differs from expected 3:4. Detection may be unstable."
        )


def collect_model_paths(model_dir: str, explicit_models: Optional[List[str]]) -> List[str]:
    if explicit_models:
        return [os.path.abspath(p) for p in explicit_models]

    pattern = os.path.join(model_dir, "*.onnx")
    model_paths = sorted(glob.glob(pattern))
    if not model_paths:
        raise FileNotFoundError(
            f"No ONNX models found in {os.path.abspath(model_dir)}. "
            "Convert your .pth checkpoints or pass explicit paths via --models."
        )
    return model_paths


def prepare_sessions(model_paths: Sequence[str], providers: Sequence[str]) -> List[Tuple[str, ort.InferenceSession]]:
    sessions: List[Tuple[str, ort.InferenceSession]] = []
    for path in model_paths:
        sessions.append((path, ort.InferenceSession(path, providers=providers)))
    return sessions


def preprocess_image(image: np.ndarray, bbox: list[int], scale: Optional[float], out_h: int, out_w: int,
                     cropper: CropImage) -> np.ndarray:
    crop_params = {
        "org_img": image,
        "bbox": bbox,
        "scale": scale,
        "out_w": out_w,
        "out_h": out_h,
        "crop": scale is not None,
    }
    resized = cropper.crop(**crop_params)
    return resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]


def compare_models(image_path: str, model_paths: Sequence[str], use_gpu: bool, save_vis: bool, output_dir: str) -> None:
    detector = Detection()
    cropper = CropImage()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    sessions = prepare_sessions(model_paths, providers)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")
    check_image_ratio(image)

    bbox = detector.get_bbox(image)
    if not bbox:
        raise RuntimeError("Face detector did not return a bounding box.")

    prediction_sum = np.zeros((1, 3), dtype=np.float32)
    per_model_results = []

    for model_path, session in sessions:
        h_input, w_input, _, scale = parse_model_name(os.path.basename(model_path))
        input_blob = preprocess_image(image, bbox, scale, h_input, w_input, cropper)

        start = time.time()
        ort_inputs = {session.get_inputs()[0].name: input_blob}
        raw_output = session.run(None, ort_inputs)[0]
        elapsed = time.time() - start

        probs = softmax(raw_output)
        prediction_sum += probs
        per_model_results.append((model_path, float(probs[0][0]), float(probs[0][1]), float(probs[0][2]), elapsed))

    avg_prediction = prediction_sum / len(sessions)
    label = int(np.argmax(avg_prediction))
    score = float(avg_prediction[0][label])
    verdict = "Real Face" if label == 1 else "Fake Face"

    print(f"Processed image: {image_path}")
    for path, p0, p1, p2, cost in per_model_results:
        print(f"- {os.path.basename(path)} -> [spoof:{p0:.3f}, real:{p1:.3f}, unsure:{p2:.3f}] "
              f"inference {cost*1000:.1f} ms")
    print(f"Average verdict: {verdict} (score={score:.3f}) using {len(sessions)} model(s)")

    if save_vis:
        annotated = image.copy()
        color = (255, 0, 0) if label == 1 else (0, 0, 255)
        cv2.rectangle(
            annotated,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            color,
            2,
        )
        cv2.putText(
            annotated,
            f"{verdict} {score:.3f}",
            (bbox[0], bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5 * image.shape[0] / 1024,
            color,
        )
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{base}_onnx_result.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"Saved annotated result to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ONNX anti-spoofing models on a single image."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./images/sample/image_F1.jpg",
        help="Path to the image to test.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/onnx_models",
        help="Directory containing ONNX models. Ignored when --models is given.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Explicit ONNX model paths. Use this to mix models from different folders.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use CUDAExecutionProvider if available.",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save an annotated image with the averaged verdict.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./images/sample",
        help="Folder for annotated results when --save_vis is enabled.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_paths = collect_model_paths(args.model_dir, args.models)
    compare_models(
        image_path=args.image,
        model_paths=model_paths,
        use_gpu=args.use_gpu,
        save_vis=args.save_vis,
        output_dir=args.output_dir,
    )
