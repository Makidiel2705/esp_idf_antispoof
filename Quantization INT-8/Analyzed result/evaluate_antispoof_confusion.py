import argparse
import contextlib
import csv
import importlib
import importlib.util
import io
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import compare_antispoof_backends as cab


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(ROOT, "raw_captures")
DEFAULT_TFLITE = os.path.join(ROOT, "antispoof.fullint8.local.tflite")
DEFAULT_ONNX = os.path.join(ROOT, "data", "antispoof.onnx")
DEFAULT_PTH = os.path.join(
    os.path.dirname(ROOT),
    "Silent-Face-Anti-Spoofing-master",
    "resources",
    "anti_spoof_models",
    "2.7_80x80_MiniFASNetV2.pth",
)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MODEL_INPUT_SIZE = 80
ANTISPOOF_CROP_SCALE = 2.7


def compute_scaled_crop(
    src_w: int,
    src_h: int,
    box_x: int,
    box_y: int,
    box_w: int,
    box_h: int,
    scale: float = ANTISPOOF_CROP_SCALE,
) -> Tuple[int, int, int, int, float]:
    if src_w <= 1 or src_h <= 1 or box_w <= 0 or box_h <= 0:
        return 0, 0, src_w, src_h, 1.0

    safe_scale = min((src_w - 1.0) / box_w, (src_h - 1.0) / box_h, scale)
    new_w = box_w * safe_scale
    new_h = box_h * safe_scale
    cx = box_x + box_w / 2.0
    cy = box_y + box_h / 2.0

    left = int(round(cx - new_w / 2.0))
    top = int(round(cy - new_h / 2.0))
    right = int(round(cx + new_w / 2.0))
    bottom = int(round(cy + new_h / 2.0))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w:
        left -= right - src_w
        right = src_w
    if bottom > src_h:
        top -= bottom - src_h
        bottom = src_h

    left = max(0, left)
    top = max(0, top)
    right = min(src_w, max(left + 1, right))
    bottom = min(src_h, max(top + 1, bottom))
    return left, top, right, bottom, safe_scale


def detect_face_opencv(rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    try:
        import cv2
    except ModuleNotFoundError:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None

    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
    if len(faces) == 0:
        return None

    # Use the largest face, matching the single-person test setup.
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(w), int(h)


def center_crop_box(src_w: int, src_h: int) -> Tuple[int, int, int, int, float]:
    side = min(src_w, src_h)
    left = (src_w - side) // 2
    top = (src_h - side) // 2
    return left, top, left + side, top + side, 1.0


def preprocess_image_to_patch(image_path: str, processed_dir: Optional[str]) -> Tuple[np.ndarray, Dict[str, object]]:
    image = Image.open(image_path).convert("RGB")
    src_w, src_h = image.size
    rgb = np.asarray(image, dtype=np.uint8)

    method = "resize_existing_patch"
    face_box = ""
    crop_left, crop_top, crop_right, crop_bottom, applied_scale = 0, 0, src_w, src_h, 1.0

    if src_w == MODEL_INPUT_SIZE and src_h == MODEL_INPUT_SIZE:
        patch = image
    else:
        face = detect_face_opencv(rgb)
        if face:
            x, y, w, h = face
            face_box = f"{x},{y},{w},{h}"
            crop_left, crop_top, crop_right, crop_bottom, applied_scale = compute_scaled_crop(src_w, src_h, x, y, w, h)
            method = "opencv_face_2.7x_crop"
        else:
            crop_left, crop_top, crop_right, crop_bottom, applied_scale = center_crop_box(src_w, src_h)
            method = "center_crop_fallback"

        patch = image.crop((crop_left, crop_top, crop_right, crop_bottom)).resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            Image.BILINEAR,
        )

    processed_path = ""
    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)
        base, _ = os.path.splitext(os.path.basename(image_path))
        processed_path = os.path.join(processed_dir, f"{base}_80x80.jpg")
        patch.save(processed_path, quality=95)

    patch_rgb = np.asarray(patch, dtype=np.float32)
    patch_bgr = patch_rgb[..., ::-1]
    meta = {
        "source_width": src_w,
        "source_height": src_h,
        "preprocess_method": method,
        "face_box_xywh": face_box,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_right": crop_right,
        "crop_bottom": crop_bottom,
        "applied_scale": applied_scale,
        "processed_path": processed_path,
    }
    return patch_bgr, meta


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTS)
    )


def choose_images(paths: List[str], count: int, seed: int) -> List[str]:
    if len(paths) < count:
        raise RuntimeError(f"Need {count} images, but only found {len(paths)} in {os.path.dirname(paths[0]) if paths else 'folder'}")
    rng = random.Random(seed)
    return sorted(rng.sample(paths, count))


def build_runner(args) -> Tuple[str, Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    backend = args.backend.lower()

    candidates = []
    if backend in ("auto", "tflite"):
        dep = "tflite_runtime" if importlib.util.find_spec("tflite_runtime") is not None else "tensorflow"
        candidates.append(("tflite_int8", args.tflite, dep, lambda p: cab.run_tflite(args.tflite, p)))
    if backend in ("auto", "onnx"):
        candidates.append(("onnx", args.onnx, "onnxruntime", lambda p: cab.run_onnx(args.onnx, p)))
    if backend in ("auto", "pytorch"):
        candidates.append(("pytorch", args.pth, "torch", lambda p: cab.run_pytorch(args.pth, p)))

    errors = []
    for name, model_path, dependency, fn in candidates:
        if not os.path.exists(model_path):
            errors.append(f"{name}: model not found: {model_path}")
            continue
        try:
            if dependency == "tensorflow":
                import tensorflow
            elif dependency == "tflite_runtime":
                import tflite_runtime
            elif dependency == "onnxruntime":
                import onnxruntime
            elif dependency == "torch":
                import torch
            else:
                importlib.import_module(dependency)
        except ImportError as e:
            errors.append(f"{name}: missing Python dependency: {dependency} (Reason: {e})")
            continue

        def run_silent(patch_bgr: np.ndarray, raw_fn=fn):
            # The shared backend prints metadata for every TFLite invoke. Keep the
            # evaluation CSV clean by silencing those per-image debug messages.
            with contextlib.redirect_stdout(io.StringIO()):
                return raw_fn(patch_bgr)

        return name, run_silent

    raise RuntimeError("No usable backend found.\n" + "\n".join(errors))


def evaluate_image(
    image_path: str,
    runner: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    threshold: float,
    processed_dir: Optional[str],
) -> Dict[str, object]:
    patch_bgr, preprocess_meta = preprocess_image_to_patch(image_path, processed_dir)
    logits, probs = runner(patch_bgr)
    logits = np.asarray(logits).reshape(-1)
    probs = np.asarray(probs).reshape(-1)

    class0 = float(probs[0]) if probs.size > 0 else float("nan")
    real = float(probs[1]) if probs.size > 1 else float("nan")
    class2 = float(probs[2]) if probs.size > 2 else float("nan")
    fake_total = 1.0 - real if probs.size > 1 else float("nan")

    if real >= threshold:
        pred_class = 0  # real
        pred_label = "real"
    else:
        pred_class = 1  # fake
        pred_label = "fake"

    row = {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        "pred_label": pred_label,
        "pred_class": pred_class,
        "class0_percent": class0 * 100.0,
        "real_percent": real * 100.0,
        "class2_percent": class2 * 100.0,
        "fake_total_percent": fake_total * 100.0,
        "logits": "[" + ", ".join(f"{x:.6f}" for x in logits.tolist()) + "]",
    }
    row.update(preprocess_meta)
    return row


def draw_confusion_matrix(matrix: np.ndarray, classes: List[str], out_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float64), where=row_sums>0)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xticks(np.arange(norm_matrix.shape[1]))
    ax.set_yticks(np.arange(norm_matrix.shape[0]))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    
    ax.set_title('Normalized confusion matrix', fontsize=12, pad=10)
    ax.set_ylabel('True label', fontsize=11, labelpad=5)
    ax.set_xlabel('Predicted label', fontsize=11, labelpad=5)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fmt = '.2f'
    thresh = norm_matrix.max() / 2.
    for i in range(norm_matrix.shape[0]):
        for j in range(norm_matrix.shape[1]):
            color = "white" if norm_matrix[i, j] > thresh else "black"
            ax.text(j, i, format(norm_matrix[i, j], fmt),
                    ha="center", va="center",
                    color=color, fontsize=10)
            
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def write_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    fieldnames = [
        "index",
        "true_label",
        "pred_label",
        "correct",
        "image_name",
        "image_path",
        "source_width",
        "source_height",
        "preprocess_method",
        "face_box_xywh",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
        "applied_scale",
        "processed_path",
        "class0_percent",
        "real_percent",
        "class2_percent",
        "fake_total_percent",
        "logits",
    ]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted.pop("pred_class", None)
            for key in ("class0_percent", "real_percent", "class2_percent", "fake_total_percent"):
                formatted[key] = f"{float(formatted[key]):.4f}"
            formatted["applied_scale"] = f"{float(formatted['applied_scale']):.4f}"
            writer.writerow(formatted)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate 100 real + 100 fake anti-spoof images and export CSV/confusion matrix.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET_DIR, help="Dataset folder containing real/ and fake/")
    ap.add_argument("--count", type=int, default=100, help="Number of images per class")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    ap.add_argument("--threshold", type=float, default=0.5, help="REAL threshold. real_prob >= threshold => predicted real")
    ap.add_argument("--backend", choices=["auto", "tflite", "onnx", "pytorch"], default="tflite")
    ap.add_argument("--tflite", default=DEFAULT_TFLITE)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--pth", default=DEFAULT_PTH)
    ap.add_argument("--csv", default=os.path.join(DEFAULT_DATASET_DIR, "antispoof_200_report.csv"))
    ap.add_argument("--matrix", default=os.path.join(DEFAULT_DATASET_DIR, "antispoof_confusion_matrix.png"))
    ap.add_argument(
        "--processed-dir",
        default=os.path.join(DEFAULT_DATASET_DIR, "processed_80x80"),
        help="Folder for preprocessed 80x80 patches. Use empty string to disable saving patches.",
    )
    args = ap.parse_args()

    real_dir = os.path.join(args.dataset, "real")
    fake_dir = os.path.join(args.dataset, "fake")
    real_images = choose_images(list_images(real_dir), args.count, args.seed)
    fake_images = choose_images(list_images(fake_dir), args.count, args.seed + 1)

    backend_name, runner = build_runner(args)
    processed_dir = args.processed_dir or None
    print(f"Backend: {backend_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Images: {len(real_images)} real + {len(fake_images)} fake")
    if processed_dir:
        print(f"Processed patches: {processed_dir}")

    rows: List[Dict[str, object]] = []
    matrix_2x2 = np.zeros((2, 2), dtype=np.int32)

    samples = [("real", path) for path in real_images] + [("fake", path) for path in fake_images]
    for idx, (true_label, image_path) in enumerate(samples, start=1):
        class_processed_dir = os.path.join(processed_dir, true_label) if processed_dir else None
        result = evaluate_image(image_path, runner, args.threshold, class_processed_dir)
        
        pred_class = int(result["pred_class"])
        true_class = 0 if true_label == "real" else 1
                
        correct = pred_class == true_class
        matrix_2x2[true_class, pred_class] += 1
        
        result.update({
            "index": idx,
            "true_label": true_label,
            "pred_label": result["pred_label"],
            "correct": "yes" if correct else "no",
        })
        rows.append(result)
        print(
            f"[{idx:03d}/{len(samples)}] true={true_label:<4} pred={result['pred_label']:<4} "
            f"real={result['real_percent']:.2f}% prep={result['preprocess_method']} "
            f"scale={result['applied_scale']:.2f} file={result['image_name']}"
        )

    total = len(rows)
    correct_count = sum(1 for row in rows if row["correct"] == "yes")
    accuracy = correct_count / total if total else 0.0

    print("\nSummary")
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct_count}/{total})")
    print(f"Confusion matrix:\n{matrix_2x2}")

    try:
        write_csv(rows, args.csv)
        print(f"CSV exported successfully: {args.csv}")
    except PermissionError:
        print(f"\n[WARNING] Permission denied writing to CSV: {args.csv}")
        print("Please close the CSV file if it is open in Microsoft Excel or another program, then run the script again.")

    try:
        draw_confusion_matrix(matrix_2x2, ['real', 'fake'], args.matrix)
        print(f"Confusion matrix image saved: {args.matrix}")
    except PermissionError:
        print(f"\n[WARNING] Permission denied saving confusion matrix: {args.matrix}")
        print("Please close the image file if it is open and run the script again.")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as e:
        print(f"\n[ERROR] Da xay ra loi: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    input("\nNhan Enter de ket thuc chuong trinh...")
    import sys
    sys.exit(exit_code)
