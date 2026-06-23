import argparse
import csv
import glob
import os
from typing import Callable, Dict, List, Tuple

import numpy as np

import compare_antispoof_backends as cab


def iter_images(folder: str) -> List[str]:
    paths: List[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def backend_runner(name: str, fn: Callable, *args):
    def run(patch_bgr: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray]:
        logits, probs = fn(*args, patch_bgr)
        return name, np.asarray(logits).reshape(-1), np.asarray(probs).reshape(-1)

    run.__name__ = name
    return run


def build_backends(args) -> List[Callable[[np.ndarray], Tuple[str, np.ndarray, np.ndarray]]]:
    backends = []
    if args.pth and os.path.exists(args.pth):
        backends.append(backend_runner("pytorch", cab.run_pytorch, args.pth))
    if args.onnx and os.path.exists(args.onnx):
        backends.append(backend_runner("onnx", cab.run_onnx, args.onnx))
    if args.tflite and os.path.exists(args.tflite):
        backends.append(backend_runner("tflite_int8", cab.run_tflite, args.tflite))
    if args.tflite_float and os.path.exists(args.tflite_float):
        backends.append(backend_runner("tflite_float", cab.run_tflite, args.tflite_float))
    return backends


def row_from_result(image_path: str, backend_name: str, logits: np.ndarray, probs: np.ndarray) -> Dict[str, str]:
    real = float(probs[1]) if probs.size > 1 else float("nan")
    return {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        f"{backend_name}_real_percent": f"{real * 100.0:.2f}",
    }


def merge_rows(base: Dict[str, str], extra: Dict[str, str]) -> Dict[str, str]:
    merged = dict(base)
    merged.update(extra)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder containing cropped face images")
    ap.add_argument("--out", default="antispoof_report.csv", help="Output CSV path")
    ap.add_argument(
        "--pth",
        default=os.path.join(os.path.dirname(__file__), "..", "Silent-Face-Anti-Spoofing-master", "resources", "anti_spoof_models", "2.7_80x80_MiniFASNetV2.pth"),
        help="Path to .pth checkpoint",
    )
    ap.add_argument(
        "--onnx",
        default=os.path.join(os.path.dirname(__file__), "data", "antispoof.legacy.onnx"),
        help="Path to ONNX model",
    )
    ap.add_argument(
        "--tflite",
        default=os.path.join(os.path.dirname(__file__), "antispoof.fullint8.local.tflite"),
        help="Path to TFLite INT8 model",
    )
    ap.add_argument(
        "--tflite-float",
        default=os.path.join(os.path.dirname(__file__), "antispoof.float32.legacy.sim.tflite"),
        help="Path to TFLite float32 model",
    )
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    out_path = os.path.abspath(args.out)
    images = iter_images(folder)
    if not images:
        raise SystemExit(f"No images found in: {folder}")

    backends = build_backends(args)
    if not backends:
        raise SystemExit("No valid backend model paths found.")

    rows: List[Dict[str, str]] = []
    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {os.path.basename(image_path)}")
        patch_bgr = cab.load_patch_bgr_255(image_path, 80)
        row: Dict[str, str] = {
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
        }

        for run_backend in backends:
            try:
                backend_name, logits, probs = run_backend(patch_bgr)
                row = merge_rows(row, row_from_result(image_path, backend_name, logits, probs))
            except ModuleNotFoundError as e:
                row[f"{run_backend.__name__}_error"] = f"missing dependency: {e}"
            except Exception as e:
                # Keep processing the rest of the folder.
                backend_name = getattr(run_backend, "__name__", "backend")
                row[f"{backend_name}_error"] = f"{type(e).__name__}: {e}"

        rows.append(row)

    fieldnames: List[str] = ["image_name", "image_path"]
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV: {out_path}")
    print("Open this file directly with Excel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
