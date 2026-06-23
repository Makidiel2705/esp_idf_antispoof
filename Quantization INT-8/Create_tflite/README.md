# Local TFLite (Full INT8) build folder

This folder is intended to run the conversion locally on Windows, installing all Python dependencies inside this folder (via a virtualenv).

## Hard constraint (Python version)

Your machine currently has only Python 3.13 (`py -0p` shows 3.13). TensorFlow and some ONNX tooling often do not provide stable wheels for Python 3.13.

To run locally, install Python 3.12 and ensure `py -3.12` exists, then use the commands below.

## Inputs (already in this repo)

- ONNX model: `D:\Project_TotNghiep\esp_idf_antispoof\data\antispoof.onnx`
  - plus `antispoof.onnx.data` in the same folder
- Calibration images:
  - Cropped faces: `D:\Project_TotNghiep\esp_idf_antispoof\raw_captures\*.jpg`
    - These are already `crop_face_*.jpg` produced from your ESP32 capture pipeline.

## Steps (PowerShell)

From `D:\Project_TotNghiep\esp_idf_antispoof`:

```powershell
py -3.12 -m venv .\Create_tflite\.venv
.\Create_tflite\.venv\Scripts\python -m pip install -U pip
.\Create_tflite\.venv\Scripts\pip install -r .\Create_tflite\requirements-py312.txt

.\Create_tflite\.venv\Scripts\python .\Create_tflite\convert_onnx_to_tflite_fullint8.py `
  --onnx ..\data\antispoof.onnx `
  --calib_dir ..\raw_captures `
  --out ..\antispoof.fullint8.tflite
```

Notes:
- `onnx2tf` currently needs `TF_USE_LEGACY_KERAS=1` with TensorFlow 2.16 / Keras 3.
- If your environment is offline, set `ONNX2TF_OFFLINE=1` to skip external asset downloads.
- The script auto-patches Conv `kernel_shape` attributes (some PyTorch exports omit them).

## Expected verification output

The script prints:

- input/output dtype (`int8`)
- input/output quantization params
- number of float tensors (should be 0)
- a warning if input scale is close to `1/255` (usually indicates wrong calibration domain)

If you want Silent-Face compatibility, calibration should be in `BGR` and in the `0..255` float domain (this is what the script does by default).
