"""
Model evaluation script:
- Computes accuracy, precision, recall, F1 on test set
- Generates confusion matrix
- Per-class performance report
- Inference speed benchmark
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, top_k_accuracy_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import config as cfg
from model.data_loader import SequenceDataGenerator, load_manifest


def load_model_for_eval(model_path=None):
    path = model_path or cfg.BEST_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    print(f"[LOAD] Loading model: {path}")
    return tf.keras.models.load_model(path)


def run_evaluation(model, label_map, batch_size=4):
    index_to_label = {v: k for k, v in label_map.items()}
    num_classes     = len(label_map)

    test_gen = SequenceDataGenerator(
        cfg.TEST_MANIFEST, label_map,
        batch_size = batch_size,
        shuffle    = False,
        augment    = False
    )

    all_preds      = []
    all_labels     = []
    all_probs      = []
    inference_times = []

    print(f"\n[EVAL] Running inference on {len(test_gen)} batches...")

    for batch_X, batch_y in test_gen:
        t_start = time.perf_counter()
        probs   = model.predict(batch_X, verbose=0)
        t_end   = time.perf_counter()

        batch_size_actual = batch_X.shape[0]
        inference_times.append((t_end - t_start) / batch_size_actual)

        preds  = np.argmax(probs, axis=1)
        labels = np.argmax(batch_y, axis=1)

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    return all_preds, all_labels, all_probs, inference_times


def generate_confusion_matrix_plot(y_true, y_pred, label_names, save_path):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

    n = len(label_names)
    figsize = max(10, n * 0.5)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.8))

    sns.heatmap(cm_pct, annot=(n <= 20), fmt='.1f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label',      fontsize=12)
    ax.set_title('Confusion Matrix (%)', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Confusion matrix saved: {save_path}")


def generate_per_class_report(y_true, y_pred, index_to_label, save_path):
    label_names = [index_to_label[i] for i in range(len(index_to_label))]
    report_str  = classification_report(y_true, y_pred,
                                         target_names=label_names,
                                         digits=4)
    report_dict = classification_report(y_true, y_pred,
                                         target_names=label_names,
                                         output_dict=True)

    print("\n" + "=" * 60)
    print("  PER-CLASS CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_str)

    with open(save_path, "w") as f:
        f.write(report_str)

    return report_dict


def benchmark_inference_speed(model, n_runs=50, batch_size=1):
    """Benchmark single-sample inference latency."""
    dummy = np.random.rand(
        batch_size,
        cfg.SEQUENCE_LENGTH, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.CHANNELS
    ).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = model.predict(dummy, verbose=0)

    times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        _ = model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - t) * 1000)  # ms

    times = np.array(times)
    print(f"\n[BENCHMARK] Inference speed ({n_runs} runs, batch={batch_size}):")
    print(f"  Mean    : {times.mean():.2f} ms")
    print(f"  Median  : {np.median(times):.2f} ms")
    print(f"  P95     : {np.percentile(times, 95):.2f} ms")
    print(f"  Min/Max : {times.min():.2f} / {times.max():.2f} ms")
    return times


def main():
    print("=" * 60)
    print("  MODEL EVALUATOR")
    print("=" * 60)

    label_map       = cfg.load_label_map()
    index_to_label  = {v: k for k, v in label_map.items()}
    label_names     = [index_to_label[i] for i in range(len(index_to_label))]

    model = load_model_for_eval()

    # Run evaluation
    y_pred, y_true, y_probs, inf_times = run_evaluation(
        model, label_map, batch_size=4
    )

    # Metrics
    accuracy  = accuracy_score(y_true, y_pred)
    top3_acc  = top_k_accuracy_score(y_true, y_probs, k=3)
    mean_inf  = np.mean(inf_times) * 1000  # ms

    print(f"\n[RESULTS]")
    print(f"  Accuracy (Top-1) : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Accuracy (Top-3) : {top3_acc:.4f} ({top3_acc*100:.2f}%)")
    print(f"  Mean inference   : {mean_inf:.2f} ms/sample")

    # Save outputs
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    generate_confusion_matrix_plot(
        y_true, y_pred, label_names,
        os.path.join(cfg.LOG_DIR, "confusion_matrix.png")
    )

    report_dict = generate_per_class_report(
        y_true, y_pred, index_to_label,
        os.path.join(cfg.LOG_DIR, "classification_report.txt")
    )

    # Benchmark
    benchmark_inference_speed(model, n_runs=30)

    # Save summary
    eval_summary = {
        "top1_accuracy":      accuracy,
        "top3_accuracy":      top3_acc,
        "mean_inference_ms":  mean_inf,
        "num_test_samples":   len(y_true),
        "num_classes":        len(label_map),
    }
    with open(os.path.join(cfg.LOG_DIR, "eval_summary.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\n[DONE] Evaluation complete. Logs saved to {cfg.LOG_DIR}")


if __name__ == "__main__":
    main()