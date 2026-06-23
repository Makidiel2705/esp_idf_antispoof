import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


ROOT = os.path.dirname(os.path.abspath(__file__))
SILENT_FACE_ROOT = os.path.join(os.path.dirname(ROOT), "Silent-Face-Anti-Spoofing-master")


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    denom = np.sum(exp)
    return exp / denom if denom > 0 else np.zeros_like(exp)


def load_patch_bgr_255(image_path: str, size: int = 80) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((size, size))
    rgb = np.asarray(img, dtype=np.float32)
    bgr = rgb[..., ::-1]
    return bgr


def summarize_patch(name: str, patch_bgr: np.ndarray) -> None:
    b = patch_bgr[..., 0]
    g = patch_bgr[..., 1]
    r = patch_bgr[..., 2]
    print(f"[{name}] patch shape={patch_bgr.shape} range=({patch_bgr.min():.1f}, {patch_bgr.max():.1f})")
    print(
        f"[{name}] mean B/G/R={b.mean():.2f}/{g.mean():.2f}/{r.mean():.2f} "
        f"std B/G/R={b.std():.2f}/{g.std():.2f}/{r.std():.2f}"
    )
    flat = patch_bgr.reshape(-1, 3)
    print(f"[{name}] first pixels BGR={flat[:5].astype(np.int32).tolist()}")


def print_result(name: str, logits: np.ndarray, probs: np.ndarray) -> None:
    logits = np.asarray(logits).reshape(-1)
    probs = np.asarray(probs).reshape(-1)
    real = probs[1] if probs.size > 1 else float("nan")
    fake_total = 1.0 - real if probs.size > 1 else float("nan")
    logits_txt = ", ".join(f"{x:.6f}" for x in logits.tolist())
    probs_txt = ", ".join(f"{x * 100:.2f}%" for x in probs.tolist())
    print(f"[{name}] logits=[{logits_txt}]")
    print(f"[{name}] probs=[{probs_txt}] real(class1)={real * 100:.2f}% fake(total)={fake_total * 100:.2f}%")


def run_pytorch(pth_path: str, patch_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    if SILENT_FACE_ROOT not in sys.path:
        sys.path.insert(0, SILENT_FACE_ROOT)

    from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
    from src.utility import get_kernel, parse_model_name

    model_map = {
        "MiniFASNetV1": MiniFASNetV1,
        "MiniFASNetV2": MiniFASNetV2,
        "MiniFASNetV1SE": MiniFASNetV1SE,
        "MiniFASNetV2SE": MiniFASNetV2SE,
    }

    ckpt_name = os.path.basename(pth_path)
    h_input, w_input, model_type, _scale = parse_model_name(ckpt_name)
    kernel_size = get_kernel(h_input, w_input)
    model = model_map[model_type](conv6_kernel=kernel_size)

    state_dict = torch.load(pth_path, map_location="cpu")
    if len(state_dict) > 0:
        first_key = next(iter(state_dict))
        if first_key.startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    chw = np.transpose(patch_bgr, (2, 0, 1)).astype(np.float32)
    batch = np.expand_dims(chw, axis=0)

    with torch.no_grad():
        out = model(torch.from_numpy(batch))
    logits = out.detach().cpu().numpy().reshape(-1)
    probs = softmax(logits)
    return logits, probs


def run_onnx(onnx_path: str, patch_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    batch = np.expand_dims(np.transpose(patch_bgr, (2, 0, 1)).astype(np.float32), axis=0)
    outputs = session.run(None, {input_name: batch})
    logits = np.asarray(outputs[0]).reshape(-1)
    probs = softmax(logits)
    return logits, probs


def run_tflite(tflite_path: str, patch_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import tflite_runtime.interpreter as tflite
        inter = tflite.Interpreter(model_path=tflite_path)
    except ModuleNotFoundError:
        import tensorflow as tf
        inter = tf.lite.Interpreter(model_path=tflite_path)

    inter.allocate_tensors()
    inp = inter.get_input_details()[0]
    out = inter.get_output_details()[0]

    scale, zero_point = inp["quantization"]
    val = patch_bgr / 255.0 if (0.0 < scale < 0.01) else patch_bgr.astype(np.float32)

    if inp["dtype"] == np.int8:
        if scale == 0:
            raise RuntimeError("TFLite input scale is 0")
        quant = np.round(val / scale + zero_point)
        quant = np.clip(quant, -128, 127).astype(np.int8)
        tensor = np.expand_dims(quant, axis=0)
    elif inp["dtype"] == np.float32:
        tensor = np.expand_dims(val, axis=0).astype(np.float32)
    else:
        raise RuntimeError(f"Unsupported TFLite input dtype: {inp['dtype']}")

    inter.set_tensor(inp["index"], tensor)
    inter.invoke()
    raw_out = inter.get_tensor(out["index"]).reshape(-1)

    if out["dtype"] == np.int8:
        out_scale, out_zero = out["quantization"]
        logits = (raw_out.astype(np.float32) - out_zero) * out_scale
    elif out["dtype"] == np.float32:
        logits = raw_out.astype(np.float32)
    else:
        raise RuntimeError(f"Unsupported TFLite output dtype: {out['dtype']}")

    probs = softmax(logits)

    print(
        f"[tflite-meta] input_dtype={inp['dtype']} input_quant={inp['quantization']} "
        f"output_dtype={out['dtype']} output_quant={out['quantization']}"
    )
    if inp["dtype"] == np.int8:
        flat = tensor.reshape(-1)
        print(f"[tflite-meta] first quantized bytes={flat[:10].astype(np.int32).tolist()}")

    return logits, probs


def try_backend(label: str, fn, *args):
    try:
        logits, probs = fn(*args)
        print_result(label, logits, probs)
        return True
    except ModuleNotFoundError as e:
        print(f"[{label}] skipped: missing dependency: {e}")
        return False
    except Exception as e:
        print(f"[{label}] failed: {type(e).__name__}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to a cropped 80x80 face patch")
    ap.add_argument(
        "--pth",
        default=os.path.join(SILENT_FACE_ROOT, "resources", "anti_spoof_models", "2.7_80x80_MiniFASNetV2.pth"),
        help="Path to .pth checkpoint",
    )
    ap.add_argument(
        "--onnx",
        default=os.path.join(ROOT, "data", "antispoof.onnx"),
        help="Path to ONNX model",
    )
    ap.add_argument(
        "--tflite",
        default=os.path.join(ROOT, "antispoof.fullint8.local.tflite"),
        help="Path to TFLite INT8 model",
    )
    ap.add_argument(
        "--tflite-float",
        default=os.path.join(ROOT, "antispoof.float32.local.tflite"),
        help="Path to TFLite float32 model",
    )
    args = ap.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"Image not found: {args.image}")

    patch_bgr = load_patch_bgr_255(args.image, 80)
    summarize_patch("input", patch_bgr)

    ok = False
    if os.path.exists(args.pth):
        ok = try_backend("pytorch", run_pytorch, args.pth, patch_bgr) or ok
    else:
        print(f"[pytorch] skipped: file not found: {args.pth}")

    if os.path.exists(args.onnx):
        ok = try_backend("onnx", run_onnx, args.onnx, patch_bgr) or ok
    else:
        print(f"[onnx] skipped: file not found: {args.onnx}")

    if os.path.exists(args.tflite):
        ok = try_backend("tflite-int8", run_tflite, args.tflite, patch_bgr) or ok
    else:
        print(f"[tflite-int8] skipped: file not found: {args.tflite}")

    if os.path.exists(args.tflite_float):
        ok = try_backend("tflite-float", run_tflite, args.tflite_float, patch_bgr) or ok
    else:
        print(f"[tflite-float] skipped: file not found: {args.tflite_float}")

    if not ok:
        print("No backend ran successfully.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
