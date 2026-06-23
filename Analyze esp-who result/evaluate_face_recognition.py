import argparse
import csv
import io
import json
import os
import random
import time
import shutil
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
NO_FACE = "__no_face__"
STRANGER = "__stranger__"
OTHER = "__other__"


def detect_face_box_pil_image(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError:
        return None

    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(w), int(h)


def opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def crop_around_box(img: Image.Image, box: Tuple[int, int, int, int], scale: float) -> Image.Image:
    src_w, src_h = img.size
    x, y, w, h = box
    crop_w = max(w * scale, 1.0)
    crop_h = max(h * scale, 1.0)
    cx = x + w / 2.0
    cy = y + h / 2.0
    left = int(round(cx - crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))
    right = int(round(cx + crop_w / 2.0))
    bottom = int(round(cy + crop_h / 2.0))
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
    return img.crop((left, top, right, bottom))


def list_people(dataset: str) -> List[str]:
    return sorted(
        name for name in os.listdir(dataset)
        if os.path.isdir(os.path.join(dataset, name)) and not name.startswith(".")
    )


def list_images(folder: str) -> List[str]:
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTS)
    )


def image_to_jpeg_bytes(path: str, resize: Optional[Tuple[int, int]], quality: int, preprocess: str, face_scale: float) -> bytes:
    img = Image.open(path).convert("RGB")
    if preprocess == "face-center":
        face = detect_face_box_pil_image(img)
        if face:
            img = crop_around_box(img, face, face_scale)
    elif preprocess == "center-crop":
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

    if resize:
        img = img.resize(resize, Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def parse_resize(value: str) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Resize format must be WIDTHxHEIGHT, for example 320x240")
    w, h = value.lower().split("x", 1)
    return int(w), int(h)


def post_image(
    base_url: str,
    endpoint: str,
    image_path: str,
    resize: Optional[Tuple[int, int]],
    quality: int,
    timeout: int,
    preprocess: str,
    face_scale: float,
) -> Dict:
    data = image_to_jpeg_bytes(image_path, resize, quality, preprocess, face_scale)
    url = base_url.rstrip("/") + endpoint
    resp = requests.post(url, data=data, headers={"Content-Type": "image/jpeg"}, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ESP32 returned non-JSON response from {endpoint}: {resp.text[:200]}") from exc


def get_json(base_url: str, endpoint: str, timeout: int) -> Dict:
    resp = requests.get(base_url.rstrip("/") + endpoint, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def draw_confusion_matrix(labels: List[str], matrix: Dict[Tuple[str, str], int], accuracy: float, out_path: str) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    col_labels = labels + [NO_FACE, STRANGER, OTHER]
    row_labels = labels

    # Build the 2D numpy array and normalize it by row
    cm = np.zeros((len(row_labels), len(col_labels)), dtype=float)
    for r, true in enumerate(row_labels):
        row_sum = sum(matrix.get((true, pred), 0) for pred in col_labels)
        for c, pred in enumerate(col_labels):
            val = matrix.get((true, pred), 0)
            if row_sum > 0:
                cm[r, c] = val / row_sum
            else:
                cm[r, c] = 0.0

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    
    # Plot normalized confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # Set tick labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    
    # Remove prefix underscores for cleaner display in chart (e.g. __no_face__ -> no_face)
    clean_col_labels = [l.replace("__", "") for l in col_labels]
    clean_row_labels = [l.replace("__", "") for l in row_labels]
    
    ax.set_xticklabels(clean_col_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(clean_row_labels, fontsize=10)

    ax.set_title("Normalized confusion matrix", fontsize=14, pad=15)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)

    # Grid line adjustments for cell borders
    ax.set_xticks(np.arange(len(col_labels) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(row_labels) + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate cells with values
    fmt = '.2f'
    thresh = 0.5
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = cm[i, j]
            # Use white text for dark cells, black for light cells
            color = "white" if val > thresh else "black"
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center",
                    color=color, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def write_csv(rows: List[Dict], out_path: str) -> None:
    fieldnames = [
        "index",
        "true_name",
        "image_path",
        "phase",
        "status",
        "matched",
        "pred_id",
        "pred_name",
        "pred_bucket",
        "similarity",
        "correct",
        "faces",
        "width",
        "height",
        "box",
        "detect_ms",
        "recognize_ms",
        "enroll_ms",
        "total_ms",
        "esp_err",
        "error",
    ]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate ESP32 face recognition with image folders.")
    ap.add_argument("--ip", required=True, help="ESP32 IP address, for example 192.168.1.19")
    ap.add_argument("--dataset", required=True, help="Dataset folder. Each subfolder is one person name.")
    ap.add_argument("--out-dir", default="face_recognition_eval_output")
    ap.add_argument("--enroll-count", type=int, default=3, help="Images per person used for ESP32 enrollment")
    ap.add_argument("--test-count", type=int, default=0, help="Images per person used for test. 0 means all remaining images")
    ap.add_argument("--include-enroll-in-test", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resize", type=parse_resize, default=None, help="Optional resize before POST, for example 320x240")
    ap.add_argument(
        "--preprocess",
        choices=["none", "fit", "center-crop", "face-center"],
        default="face-center",
        help="Image preprocessing before POST. face-center crops around a PC-detected face first.",
    )
    ap.add_argument("--face-scale", type=float, default=2.2, help="Crop scale used by --preprocess face-center")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--timeout", type=int, default=40)
    ap.add_argument("--no-clear", action="store_true", help="Do not clear ESP32 face DB before enrollment")
    args = ap.parse_args()
    if args.resize is None and args.preprocess in ("fit", "center-crop", "face-center"):
        args.resize = (320, 240)
    if args.preprocess == "face-center" and not opencv_available():
        raise SystemExit(
            "Missing OpenCV for --preprocess face-center. Run: python -m pip install -r requirements.txt"
        )

    base_url = f"http://{args.ip}"
    people = list_people(args.dataset)
    if not people:
        raise SystemExit(f"No person folders found in {args.dataset}")

    rng = random.Random(args.seed)
    split: Dict[str, Dict[str, List[str]]] = {}
    for person in people:
        images = list_images(os.path.join(args.dataset, person))
        enroll_map = {}
        test_images = []
        for img_path in images:
            fname = os.path.basename(img_path).lower()
            name_no_ext, _ = os.path.splitext(fname)
            if fname == "1.png" or name_no_ext == "1":
                enroll_map[1] = img_path
            elif fname == "2.png" or name_no_ext == "2":
                enroll_map[2] = img_path
            elif fname == "3.png" or name_no_ext == "3":
                enroll_map[3] = img_path
            else:
                test_images.append(img_path)

        if 1 not in enroll_map or 2 not in enroll_map or 3 not in enroll_map:
            missing = [f"{i}.png" for i in (1, 2, 3) if i not in enroll_map]
            raise SystemExit(f"{person}: missing enroll images: {', '.join(missing)}")

        # Shuffle the test images to maintain some randomness as before
        shuffled_test = list(test_images)
        rng.shuffle(shuffled_test)
        
        enroll_images = [enroll_map[1], enroll_map[2], enroll_map[3]]
        split[person] = {"candidates": enroll_images + shuffled_test, "enroll": [], "test": []}

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "face_recognition_results.csv")
    matrix_path = os.path.join(args.out_dir, "face_recognition_confusion_matrix.png")

    print(f"ESP32: {base_url}")
    print(f"People: {', '.join(people)}")
    print(f"Preprocess: {args.preprocess}, resize={args.resize}, face_scale={args.face_scale}")
    print(f"Output: {os.path.abspath(args.out_dir)}")

    if not args.no_clear:
        print("[1/3] Clearing ESP32 face database...")
        print(get_json(base_url, "/clear", args.timeout))

    rows: List[Dict] = []
    print("[2/3] Enrolling faces on ESP32...")
    for person in people:
        for image_path in split[person]["candidates"]:
            if len(split[person]["enroll"]) >= args.enroll_count:
                break
            endpoint = "/enroll?name=" + quote(person)
            result = post_image(
                base_url,
                endpoint,
                image_path,
                args.resize,
                args.jpeg_quality,
                args.timeout,
                args.preprocess,
                args.face_scale,
            )
            enroll_ok = result.get("status") == "ok"
            if enroll_ok:
                split[person]["enroll"].append(image_path)
            print(
                f"  enroll person={person} image={os.path.basename(image_path)} "
                f"ok={enroll_ok} success={len(split[person]['enroll'])}/{args.enroll_count} -> {result}"
            )
            rows.append({
                "index": len(rows) + 1,
                "true_name": person,
                "image_path": image_path,
                "phase": "enroll",
                "status": result.get("status", ""),
                "matched": "",
                "pred_id": result.get("id", ""),
                "pred_name": result.get("name", ""),
                "pred_bucket": "",
                "similarity": "",
                "correct": "",
                "faces": result.get("faces", ""),
                "width": result.get("width", ""),
                "height": result.get("height", ""),
                "box": json.dumps(result.get("box", ""), ensure_ascii=False),
                "detect_ms": result.get("detect_ms", ""),
                "recognize_ms": "",
                "enroll_ms": result.get("enroll_ms", ""),
                "total_ms": "",
                "esp_err": result.get("esp_err", ""),
                "error": result.get("error", ""),
            })
            time.sleep(0.05)

        if len(split[person]["enroll"]) < args.enroll_count:
            write_csv(rows, csv_path)
            raise SystemExit(
                f"{person}: only enrolled {len(split[person]['enroll'])}/{args.enroll_count} images successfully. "
                f"Check failed enroll rows in CSV: {csv_path}"
            )

    print("  faces:", get_json(base_url, "/faces", args.timeout))

    for person in people:
        enrolled = set(split[person]["enroll"])
        if args.include_enroll_in_test:
            test_pool = list(split[person]["candidates"])
        else:
            test_pool = [path for path in split[person]["candidates"] if path not in enrolled]
        if args.test_count > 0:
            test_pool = test_pool[:args.test_count]
        if not test_pool:
            write_csv(rows, csv_path)
            raise SystemExit(f"{person}: no test images after successful enrollment. Add images or use --include-enroll-in-test")
        split[person]["test"] = test_pool

    print("[3/3] Running recognition test...")
    col_labels = people + [NO_FACE, STRANGER, OTHER]
    matrix: Dict[Tuple[str, str], int] = {(true, pred): 0 for true in people for pred in col_labels}
    test_rows = 0
    correct_rows = 0

    for person in people:
        for image_path in split[person]["test"]:
            result = post_image(
                base_url,
                "/recognize",
                image_path,
                args.resize,
                args.jpeg_quality,
                args.timeout,
                args.preprocess,
                args.face_scale,
            )
            pred_name = result.get("pred_name", "") or ""
            matched = bool(result.get("matched", False))
            faces_count = result.get("faces", 0)
            if not matched:
                if faces_count == 0:
                    pred_bucket = NO_FACE
                else:
                    pred_bucket = STRANGER
            elif pred_name in people:
                pred_bucket = pred_name
            else:
                pred_bucket = OTHER

            correct = pred_bucket == person
            matrix[(person, pred_bucket)] += 1
            test_rows += 1
            correct_rows += 1 if correct else 0

            if not correct:
                is_move = False
                if pred_bucket == NO_FACE:
                    subfolder = "no_face"
                    is_move = True
                elif pred_bucket == STRANGER:
                    subfolder = "stranger"
                    is_move = True
                else:
                    subfolder = "other"
                dest_dir = os.path.join(args.out_dir, person, subfolder)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(image_path))
                try:
                    if is_move:
                        shutil.move(image_path, dest_path)
                    else:
                        shutil.copy(image_path, dest_path)
                except Exception as e:
                    action = "move" if is_move else "copy"
                    print(f"    Failed to {action} {image_path} to {dest_path}: {e}")

            print(
                f"  test true={person:<16} pred={pred_bucket:<16} "
                f"sim={result.get('similarity', '')} image={os.path.basename(image_path)}"
            )
            rows.append({
                "index": len(rows) + 1,
                "true_name": person,
                "image_path": image_path,
                "phase": "test",
                "status": result.get("status", ""),
                "matched": matched,
                "pred_id": result.get("pred_id", ""),
                "pred_name": pred_name,
                "pred_bucket": pred_bucket,
                "similarity": result.get("similarity", ""),
                "correct": "yes" if correct else "no",
                "faces": result.get("faces", ""),
                "width": result.get("width", ""),
                "height": result.get("height", ""),
                "box": json.dumps(result.get("box", ""), ensure_ascii=False),
                "detect_ms": result.get("detect_ms", ""),
                "recognize_ms": result.get("recognize_ms", ""),
                "enroll_ms": "",
                "total_ms": result.get("total_ms", ""),
                "esp_err": result.get("esp_err", ""),
                "error": result.get("error", ""),
            })
            time.sleep(0.05)

    accuracy = correct_rows / test_rows if test_rows else 0.0
    write_csv(rows, csv_path)
    draw_confusion_matrix(people, matrix, accuracy, matrix_path)

    print("\nSummary")
    print(f"Test accuracy: {accuracy * 100:.2f}% ({correct_rows}/{test_rows})")
    print(f"CSV: {os.path.abspath(csv_path)}")
    print(f"Confusion matrix: {os.path.abspath(matrix_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
