import tensorflow as tf
import os
import sys

# Ensure output is UTF-8 to handle emojis, but we'll use ASCII for safety
model_path = r'd:\Project_TotNghiep\esp_idf_antispoof\antispoof.fullint8.tflite'

if not os.path.exists(model_path):
    print(f"Error: File not found at {model_path}")
    sys.exit(1)

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("--- Model Details ---")
    print(f"Input Name: {input_details[0]['name']}")
    print(f"Input Shape: {input_details[0]['shape']}")
    print(f"Input Dtype: {input_details[0]['dtype']}")
    print(f"Input Quantization (Scale): {input_details[0]['quantization'][0]}")
    print(f"Input Quantization (Zero Pt): {input_details[0]['quantization'][1]}")

    print("\n--- Output Details ---")
    print(f"Output Name: {output_details[0]['name']}")
    print(f"Output Shape: {output_details[0]['shape']}")
    print(f"Output Dtype: {output_details[0]['dtype']}")
    print(f"Output Quantization (Scale): {output_details[0]['quantization'][0]}")
    print(f"Output Quantization (Zero Pt): {output_details[0]['quantization'][1]}")

    # Check for "Full Integer Quantization"
    is_int8 = "int8" in str(input_details[0]['dtype']) and "int8" in str(output_details[0]['dtype'])
    
    if is_int8:
        print("\nSTATUS: Model is FULL INT8 quantized.")
    else:
        print("\nSTATUS: Model is NOT full INT8. It might be float32 or mixed.")

    # Check for ESP-IDF compatibility (ESP-DL/ESP-NN)
    print("\n--- Compatibility Check ---")
    if is_int8:
        print("- Quantization: OK for ESP32-S3 (Int8)")
        if input_details[0]['shape'][1] == 80 and input_details[0]['shape'][2] == 80:
             print("- Input Resolution: 80x80 (Standard for ESP-WHO/Antispoofing)")
    else:
        print("- Quantization: WARNING (ESP32-S3 runs Int8 models significantly faster than Float32)")

except Exception as e:
    print(f"Error loading model: {e}")
