"""
Calibration analysis for Module 7 Week A Stretch.

Builds a reliability diagram and computes Expected Calibration Error.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability.

    Returns:
      bucket_centers
      bucket_accuracies
      bucket_counts
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    edges = np.linspace(0, 1, n_bins + 1)
    bucket_centers = (edges[:-1] + edges[1:]) / 2
    bucket_accuracies = np.zeros(n_bins)
    bucket_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)

        bucket_counts[i] = int(np.sum(in_bin))

        if bucket_counts[i] > 0:
            bucket_accuracies[i] = np.mean(predictions[in_bin] == y_true[in_bin])
        else:
            bucket_accuracies[i] = 0.0

    return bucket_centers, bucket_accuracies, bucket_counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of:
      (bucket_count / N) * abs(bucket_accuracy - bucket_confidence)
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    edges = np.linspace(0, 1, n_bins + 1)
    n = len(y_true)

    ece = 0.0

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)

        count = np.sum(in_bin)

        if count == 0:
            continue

        bucket_accuracy = np.mean(predictions[in_bin] == y_true[in_bin])
        bucket_confidence = np.mean(confidences[in_bin])

        ece += (count / n) * abs(bucket_accuracy - bucket_confidence)

    return float(ece)


def plot_reliability(centers, accs, counts, output_path: str) -> None:
    """
    Save a reliability diagram.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(centers, accs, marker="o", label="Model")

    for x, y, count in zip(centers, accs, counts):
        if count > 0:
            plt.text(x, y, str(count), ha="center", va="bottom", fontsize=8)

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """
    Load manual_predictions.csv, compute reliability diagram and ECE,
    then save calibration_results.json and figures/reliability-diagram.png.
    """
    if not Path("manual_predictions.csv").exists():
        raise FileNotFoundError(
            "manual_predictions.csv was not found. Run python manual_eval.py first."
        )

    df = pd.read_csv("manual_predictions.csv")

    y_true = df["label"].map({"negative": 0, "neutral": 1, "positive": 2}).to_numpy()

    probs = df[
        [
            "manual_prob_negative",
            "manual_prob_neutral",
            "manual_prob_positive",
        ]
    ].to_numpy()

    centers, accs, counts = reliability_diagram(
        probs=probs,
        y_true=y_true,
        n_bins=10,
    )

    ece = expected_calibration_error(
        probs=probs,
        y_true=y_true,
        n_bins=10,
    )

    plot_reliability(
        centers=centers,
        accs=accs,
        counts=counts,
        output_path="figures/reliability-diagram.png",
    )

    results = {
        "ece": ece,
        "bucket_centers": centers.tolist(),
        "bucket_accuracies": accs.tolist(),
        "bucket_counts": counts.tolist(),
    }

    with open("calibration_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Calibration analysis complete.")
    print(f"ECE: {ece:.4f}")

    print("\nBuckets:")
    for center, acc, count in zip(centers, accs, counts):
        print(f"center={center:.2f}, accuracy={acc:.4f}, count={count}")


if __name__ == "__main__":
    main()