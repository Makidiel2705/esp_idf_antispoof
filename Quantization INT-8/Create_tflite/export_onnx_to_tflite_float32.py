import argparse
import os
import subprocess
import sys
import tempfile


def patch_onnx_add_kernel_shape(in_path: str) -> str:
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
        if any(a.name == "kernel_shape" for a in n.attribute):
            continue
        if len(n.input) < 2:
            continue
        w_init = init_map.get(n.input[1])
        if w_init is None:
            continue
        w_shape = list(w_init.dims)
        if len(w_shape) < 3:
            continue
        n.attribute.append(onnx.helper.make_attribute("kernel_shape", w_shape[2:]))
        changed += 1

    if changed == 0:
        return in_path

    out_path = os.path.splitext(in_path)[0] + ".kernelshape.onnx"
    onnx.save(m, out_path)
    print(f"Patched ONNX: added kernel_shape to {changed} Conv nodes -> {out_path}")
    return out_path


def verify_tflite(path: str) -> None:
    import numpy as np
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX file")
    ap.add_argument("--out", required=True, help="Output .tflite path")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        raise SystemExit(f"ONNX not found: {args.onnx}")

    onnx_path = patch_onnx_add_kernel_shape(args.onnx)
    saved_model_dir = tempfile.mkdtemp(prefix="saved_model_float_")
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
        "-osd",
        "--non_verbose",
    ]
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)

    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite = converter.convert()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(tflite)
    print("Wrote:", args.out, "bytes:", len(tflite))
    verify_tflite(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
