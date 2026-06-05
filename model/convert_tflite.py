"""
Convert trained Keras model to TensorFlow Lite for deployment.
Supports float32, float16 quantization, and INT8 quantization.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import config as cfg
from model.data_loader import SequenceDataGenerator


def convert_to_tflite(model_path, output_path, quantization="float16"):
    """
    Convert Keras .h5 model to TFLite.
    quantization: "none" | "float16" | "int8"
    """
    print(f"[CONVERT] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("[CONVERT] Applying float16 quantization")

    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Representative dataset for INT8 calibration
        label_map = cfg.load_label_map()

        def representative_dataset():
            gen = SequenceDataGenerator(
                cfg.VAL_MANIFEST, label_map,
                batch_size=1, shuffle=False, augment=False
            )
            for i, (X, _) in enumerate(gen):
                if i >= 100:  # 100 samples for calibration
                    break
                yield [X.astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8
        print("[CONVERT] Applying INT8 quantization (calibrating...)")

    else:
        print("[CONVERT] No quantization (float32)")

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    original_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[CONVERT] TFLite model saved: {output_path}")
    print(f"  Original size : {original_mb:.2f} MB")
    print(f"  TFLite size   : {size_mb:.2f} MB")
    print(f"  Compression   : {original_mb/size_mb:.1f}x")

    return output_path


def test_tflite_model(tflite_path, label_map):
    """Run a quick inference test on the TFLite model."""
    print(f"\n[TEST] Testing TFLite model: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input shape  : {input_details[0]['shape']}")
    print(f"  Output shape : {output_details[0]['shape']}")
    print(f"  Input dtype  : {input_details[0]['dtype']}")

    # Dummy input
    dummy = np.random.rand(
        1, cfg.SEQUENCE_LENGTH, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.CHANNELS
    ).astype(input_details[0]['dtype'])

    import time
    times = []
    for _ in range(10):
        t = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        times.append((time.perf_counter() - t) * 1000)

    index_to_label = {v: k for k, v in label_map.items()}
    pred_idx  = np.argmax(output[0])
    pred_label = index_to_label.get(pred_idx, "unknown")
    confidence = float(output[0][pred_idx])

    print(f"  Test prediction: {pred_label} ({confidence:.3f})")
    print(f"  Inference time : {np.mean(times):.2f} ms (mean of 10 runs)")


def main():
    print("=" * 60)
    print("  TFLITE CONVERTER")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", type=str, default="float16",
                        choices=["none", "float16", "int8"],
                        help="Quantization type")
    parser.add_argument("--model", type=str, default=cfg.BEST_MODEL_PATH,
                        help="Path to source .h5 model")
    args = parser.parse_args()

    tflite_path = cfg.TFLITE_MODEL_PATH.replace(".tflite",
                  f"_{args.quantization}.tflite")

    convert_to_tflite(args.model, tflite_path, args.quantization)

    label_map = cfg.load_label_map()
    test_tflite_model(tflite_path, label_map)

    # Save metadata alongside tflite
    meta_path = tflite_path.replace(".tflite", "_metadata.json")
    meta = {
        "model_path":    tflite_path,
        "quantization":  args.quantization,
        "input_shape":   [1, cfg.SEQUENCE_LENGTH, cfg.IMG_HEIGHT,
                          cfg.IMG_WIDTH, cfg.CHANNELS],
        "label_map":     label_map,
        "seq_length":    cfg.SEQUENCE_LENGTH,
        "img_size":      cfg.IMG_HEIGHT,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] TFLite conversion complete.")
    print(f"  Model    : {tflite_path}")
    print(f"  Metadata : {meta_path}")


if __name__ == "__main__":
    main()