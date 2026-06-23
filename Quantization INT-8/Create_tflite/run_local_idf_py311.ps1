$ErrorActionPreference = "Stop"

$idfPy = "C:\Espressif\tools\idf-python\3.11.2\python.exe"
if (!(Test-Path $idfPy)) {
  throw "ESP-IDF python not found at $idfPy"
}

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $PSScriptRoot ".venv"
$py = Join-Path $venv "Scripts\python.exe"
$pip = Join-Path $venv "Scripts\pip.exe"

if (!(Test-Path $py)) {
  & $idfPy -m venv $venv
}

& $py -m pip install -U pip
& $pip install -r (Join-Path $PSScriptRoot "requirements-idf-py311.txt")

$onnx = Join-Path $root "data\antispoof.onnx"
$calib = Join-Path $root "raw_captures"
$out = Join-Path $root "antispoof.fullint8.tflite"

& $py (Join-Path $PSScriptRoot "convert_onnx_to_tflite_fullint8.py") --onnx $onnx --calib_dir $calib --out $out

