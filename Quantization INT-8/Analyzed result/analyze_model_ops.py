"""Analyze TFLite model to count operations and estimate ESP-NN coverage."""
import sys
try:
    import tensorflow as tf
except ImportError:
    # Try tflite_runtime as fallback
    try:
        import tflite_runtime.interpreter as tflite
        USE_TFLITE_RUNTIME = True
    except ImportError:
        print("ERROR: Neither tensorflow nor tflite_runtime is installed.")
        sys.exit(1)
    USE_TFLITE_RUNTIME = False

import os

model_path = r'd:\Project_TotNghiep\esp_idf_antispoof\antispoof.fullint8.local.tflite'

if not os.path.exists(model_path):
    print(f"Error: File not found at {model_path}")
    sys.exit(1)

# Use flatbuffers to parse the model schema
try:
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get tensor details
    tensor_details = interpreter.get_tensor_details()
    
    # Get op details via the internal model object
    # Parse via flatbuffers
    from tensorflow.lite.python import schema_py_generated as schema_fb
    import flatbuffers
    
    buf = bytearray(model_data)
    model = schema_fb.Model.GetRootAs(buf, 0)
    
    subgraph = model.Subgraphs(0)
    
    # Build opcode table
    opcodes = []
    for i in range(model.OperatorCodesLength()):
        op_code = model.OperatorCodes(i)
        # BuiltinOperator enum
        builtin = op_code.DeprecatedBuiltinCode()
        if builtin == 0:
            builtin = op_code.BuiltinCode()
        opcodes.append(builtin)
    
    # Map builtin codes to names
    from tensorflow.lite.python.schema_py_generated import BuiltinOperator
    builtin_names = {}
    for attr in dir(BuiltinOperator):
        val = getattr(BuiltinOperator, attr)
        if isinstance(val, int):
            builtin_names[val] = attr
    
    # Count operations
    op_counts = {}
    total_ops = subgraph.OperatorsLength()
    
    for i in range(total_ops):
        op = subgraph.Operators(i)
        opcode_idx = op.OpcodeIndex()
        builtin_code = opcodes[opcode_idx]
        name = builtin_names.get(builtin_code, f"UNKNOWN({builtin_code})")
        op_counts[name] = op_counts.get(name, 0) + 1
    
    # ESP-NN optimized ops 
    ESP_NN_OPTIMIZED = {
        'CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED',
        'ADD', 'MUL', 'SOFTMAX', 'AVERAGE_POOL_2D', 'MAX_POOL_2D'
    }
    
    print("=" * 60)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Size: {len(model_data)} bytes ({len(model_data)/1024:.1f} KB)")
    print(f"Total operator count: {total_ops}")
    print("=" * 60)
    
    print(f"\n{'Operator':<30} {'Count':>5} {'ESP-NN Optimized?':>20}")
    print("-" * 60)
    
    optimized_count = 0
    non_optimized_count = 0
    
    for name, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        is_optimized = name in ESP_NN_OPTIMIZED
        status = "YES (HW Accel)" if is_optimized else "NO (Reference C)"
        print(f"  {name:<28} {count:>5}   {status}")
        if is_optimized:
            optimized_count += count
        else:
            non_optimized_count += count
    
    print("-" * 60)
    print(f"  ESP-NN optimized ops: {optimized_count}/{total_ops} ({100*optimized_count/total_ops:.1f}%)")
    print(f"  Unoptimized (reference C): {non_optimized_count}/{total_ops} ({100*non_optimized_count/total_ops:.1f}%)")
    
    # Specifically flag PReLU
    prelu_count = op_counts.get('PRELU', 0)
    if prelu_count > 0:
        print(f"\n{'!'*60}")
        print(f"  WARNING: {prelu_count} PReLU layers detected!")
        print(f"  PReLU uses generic `BroadcastPrelu4DSlow` reference C code.")
        print(f"  This is a 4-deep nested loop with per-element:")
        print(f"    - int32 multiply")
        print(f"    - MultiplyByQuantizedMultiplier (heavy bit-shift ops)")
        print(f"    - Conditional branch per element")
        print(f"  NO SIMD/vectorization is applied.")
        print(f"{'!'*60}")
    
    # Estimate tensor sizes for PReLU layers
    print(f"\n--- Tensor Memory Analysis ---")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"Output: {output_details[0]['shape']} ({output_details[0]['dtype']})")
    
    # Count total parameters
    total_params = 0
    for td in tensor_details:
        shape = td['shape']
        if len(shape) > 0:
            size = 1
            for s in shape:
                size *= s
            total_params += size
    print(f"Total tensor elements: {total_params:,}")

except Exception as e:
    print(f"Error analyzing model: {e}")
    import traceback
    traceback.print_exc()
