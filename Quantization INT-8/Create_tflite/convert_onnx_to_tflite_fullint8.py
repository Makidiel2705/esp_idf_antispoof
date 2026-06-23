import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image


def patch_onnx_add_kernel_shape(in_path: str) -> str:
    """
    onnx2tf expects Conv nodes to have `kernel_shape` attribute, but PyTorch ONNX
    export may omit it (it's optional per spec). Patch it in from the weight
    initializer shape: [C_OUT, C_IN, kH, kW] -> kernel_shape=[kH, kW].
    """
    try:
        import onnx
    except Exception:
        return in_path

    m = onnx.load(in_path)
    init_map = {i.name: i for i in m.graph.initializer}

    changed = 0
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        # Check if kernel_shape already exists.
        if any(a.name == "kernel_shape" for a in n.attribute):
            continue
        if len(n.input) < 2:
            continue
        w_init = init_map.get(n.input[1])
        if w_init is None:
            continue

        # weight dims: [C_OUT, C_IN, k..., k]
        w_shape = list(w_init.dims)
        if len(w_shape) < 3:
            continue
        kernel_shape = w_shape[2:]
        n.attribute.append(onnx.helper.make_attribute("kernel_shape", kernel_shape))
        changed += 1

    if changed == 0:
        return in_path

    out_path = os.path.splitext(in_path)[0] + ".kernelshape.onnx"
    onnx.save(m, out_path)
    print(f"Patched ONNX: added kernel_shape to {changed} Conv nodes -> {out_path}")
    return out_path


def load_images(calib_dir, size, limit):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob.glob(os.path.join(calib_dir, ext)))
    paths = sorted(paths)[:limit]
    if not paths:
        raise SystemExit(f"No images found in: {calib_dir}")

    samples_nhwc = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size))
        rgb = np.asarray(im, dtype=np.float32)  # 0..255
        # Silent-Face uses OpenCV-style BGR ordering. Keep float domain 0..255.
        bgr = rgb[..., ::-1]
        samples_nhwc.append(bgr)

    arr = np.stack(samples_nhwc, axis=0)  # [N,H,W,C]
    return arr


def representative_dataset(calib_nhwc):
    for i in range(calib_nhwc.shape[0]):
        yield [calib_nhwc[i : i + 1].astype(np.float32)]


def verify_tflite(path):
    import tensorflow as tf

    inter = tf.lite.Interpreter(model_path=path)
    inter.allocate_tensors()
    inp = inter.get_input_details()[0]
    out = inter.get_output_details()[0]

    print("Input shape:", inp["shape"])
    print("Input dtype:", inp["dtype"])
    print("Input quant:", inp["quantization"])
    print("Output shape:", out["shape"])
    print("Output dtype:", out["dtype"])
    print("Output quant:", out["quantization"])

    float_tensors = []
    for t in inter.get_tensor_details():
        if t["dtype"] in (np.float32, np.float16):
            float_tensors.append((t["index"], t["name"], str(t["dtype"])))
    print("Float tensors:", len(float_tensors))
    if float_tensors:
        print("First float tensors:", float_tensors[:10])

    in_scale = inp["quantization"][0]
    if in_scale and in_scale < 0.01:
        print("WARNING: input scale looks close to 1/255. This usually indicates calibration was done on 0..1 data.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX file")
    ap.add_argument("--calib_dir", required=True, help="Directory with calibration images (cropped faces recommended)")
    ap.add_argument("--out", required=True, help="Output .tflite path")
    ap.add_argument("--size", type=int, default=80, help="Model input size (default: 80)")
    ap.add_argument("--limit", type=int, default=300, help="Max calibration images (default: 300)")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        raise SystemExit(f"ONNX not found: {args.onnx}")
    if not os.path.isdir(args.calib_dir):
        raise SystemExit(f"Calibration dir not found: {args.calib_dir}")

    calib = load_images(args.calib_dir, args.size, args.limit)
    print("Loaded calib:", calib.shape, calib.dtype, float(calib.min()), float(calib.max()))

    onnx_path = patch_onnx_add_kernel_shape(args.onnx)

    # Convert ONNX -> SavedModel
    # Use module invocation to work with local --target installs (no console script needed).
    import subprocess
    import tempfile

    saved_model_dir = tempfile.mkdtemp(prefix="saved_model_")
    cmd = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        onnx_path,
        "-o",
        saved_model_dir,
        "-b",
        "1",
        "-osd",  # required so SavedModel has signatures for TFLiteConverter.from_saved_model()
        "--non_verbose",
    ]
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)

    # SavedModel -> TFLite full-int8
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(calib)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite = converter.convert()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(tflite)
    print("Wrote:", args.out, "bytes:", len(tflite))

    verify_tflite(args.out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
