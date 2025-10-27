#!/usr/bin/env python3
"""Quick test script to verify TFLite models can be loaded and run inference"""
import tensorflow as tf
import numpy as np
from pathlib import Path

# Class names (from README)
CLASS_NAMES = ["gediminas_tower", "vilnius_cathedral", "gate_of_dawn", "st_anne", "three_crosses"]
IMG_SIZE = (224, 224)

def test_tflite_model(model_path):
    """Test a TFLite model with random input"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print(f"{'='*60}")

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False

    # Get file size
    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input/output details
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        print(f"✓ Model loaded successfully")
        print(f"  Input shape: {input_details['shape']}")
        print(f"  Input dtype: {input_details['dtype']}")
        print(f"  Output shape: {output_details['shape']}")
        print(f"  Output dtype: {output_details['dtype']}")

        # Create random test input
        input_shape = input_details['shape']
        test_input = np.random.rand(*input_shape).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])

        # Handle quantized output if needed
        if output_details['dtype'] == np.int8:
            scale = output_details['quantization_parameters']['scales']
            zero_point = output_details['quantization_parameters']['zero_points']
            if len(scale) > 0:
                output = (output.astype(np.float32) - zero_point) * scale

        print(f"✓ Inference successful")
        print(f"  Output shape: {output.shape}")

        # Show top-3 predictions
        if len(output.shape) == 2 and output.shape[1] == 5:
            top3_idx = np.argsort(output[0])[-3:][::-1]
            print(f"\n  Top-3 predictions (random input):")
            for i, idx in enumerate(top3_idx, 1):
                print(f"    {i}. {CLASS_NAMES[idx]}: {output[0][idx]:.4f}")
        else:
            print(f"  Raw output: {output[0]}")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("TFLite Model Verification Test")
    print("="*60)

    models_to_test = [
        "exports/landmark_mnv3_fp32.tflite",
        "exports/landmark_mnv3_int8_drq.tflite"
    ]

    results = {}
    for model_path in models_to_test:
        results[model_path] = test_tflite_model(model_path)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for model_path, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {model_path}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All TFLite models are working correctly!")
    else:
        print(f"\n❌ Some models failed to load or run")

    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
