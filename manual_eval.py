"""
Manual evaluation for Module 7 Week A Stretch.

This file intentionally avoids Trainer.predict and sklearn metric helpers.
It uses a manual PyTorch inference loop and computes metrics from numpy arrays.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def manual_predict(model, tokenizer, texts: list[str], batch_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Manual PyTorch inference. No Trainer.predict.

    Returns:
      preds: shape (N,), int class indices
      probs: shape (N, num_classes), probabilities after softmax
    """
    model.eval()

    all_probs = []
    all_preds = []

    device = next(model.parameters()).device

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits.detach().cpu().numpy()

        probs = softmax(logits)
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(preds)

    probs = np.vstack(all_probs)
    preds = np.concatenate(all_preds)

    return preds, probs


def compute_classification_report_from_arrays(y_true, y_pred) -> dict:
    """
    Compute accuracy, per-class precision/recall/F1, and macro-F1 using numpy only.
    No sklearn metric helpers.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    accuracy = float(np.mean(y_true == y_pred))

    per_class = {}
    f1_values = []

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        per_class[int(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        f1_values.append(f1)

    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def main() -> None:
    """
    Load the local Lab 7A model, run manual predictions on predictions.csv,
    and save manual_metrics.json plus manual_predictions.csv.
    """
    model_dir = "model"

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            "model/ directory was not found. Run Lab 7A first so the fine-tuned model exists locally."
        )

    if not Path("predictions.csv").exists():
        raise FileNotFoundError(
            "predictions.csv was not found. Run Lab 7A first to create the test split artifact."
        )

    df = pd.read_csv("predictions.csv")

    texts = df["text"].tolist()
    y_true = df["label"].map({"negative": 0, "neutral": 1, "positive": 2}).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_pred, probs = manual_predict(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=8,
    )

    report = compute_classification_report_from_arrays(y_true, y_pred)

    with open("manual_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    out = df.copy()
    out["manual_predicted_label"] = [ID2LABEL[int(i)] for i in y_pred]
    out["manual_predicted_probability"] = probs.max(axis=1)

    for label_id, label_name in ID2LABEL.items():
        out[f"manual_prob_{label_name}"] = probs[:, label_id]

    out.to_csv("manual_predictions.csv", index=False)

    print("Manual evaluation complete.")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro-F1: {report['macro_f1']:.4f}")

    for label_id, values in report["per_class"].items():
        label_name = ID2LABEL[int(label_id)]
        print(
            f"{label_name}: "
            f"P={values['precision']:.4f}, "
            f"R={values['recall']:.4f}, "
            f"F1={values['f1']:.4f}"
        )


if __name__ == "__main__":
    main()