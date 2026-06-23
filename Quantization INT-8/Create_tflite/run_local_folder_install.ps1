$ErrorActionPreference = "Stop"

$idfPy = "C:\Espressif\python_env\idf5.3_py3.11_env\Scripts\python.exe"
if (!(Test-Path $idfPy)) {
  throw "Python not found at $idfPy"
}

$root = Split-Path -Parent $PSScriptRoot
$deps = Join-Path $PSScriptRoot ".pydeps"

if (Test-Path $deps) {
  try {
    Remove-Item -Recurse -Force $deps
  } catch {
    Write-Warning "Could not delete existing .pydeps (likely locked by another process). Will reuse and upgrade in place. Error: $($_.Exception.Message)"
  }
}
if (!(Test-Path $deps)) {
  New-Item -ItemType Directory -Path $deps | Out-Null
}

Write-Host "Installing deps into: $deps"
& $idfPy -m pip install --upgrade --target $deps -r (Join-Path $PSScriptRoot "requirements-idf-py311.txt")

$env:PYTHONPATH = $deps

$onnx = Join-Path $root "data\antispoof.onnx"
$calib = Join-Path $root "raw_captures"
$out = Join-Path $root "antispoof.fullint8.tflite"

& $idfPy (Join-Path $PSScriptRoot "convert_onnx_to_tflite_fullint8.py") --onnx $onnx --calib_dir $calib --out $out
