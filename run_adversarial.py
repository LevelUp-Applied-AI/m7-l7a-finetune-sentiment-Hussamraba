"""
Module 7 Week A — Stretch: Adversarial Evaluation.

Runs a fine-tuned sentiment classifier against a hand-crafted adversarial set.
The default model path is the local Lab 7A checkpoint directory: model/.

Outputs:
- results.csv
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or Hugging Face model id.

    Default:
        model/
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()

    return model, tokenizer


def normalize_label(label):
    """
    Normalize common sentiment labels.

    This keeps the code flexible because CI may use a binary public model,
    while the real Lab 7A model is usually a 3-class classifier.
    """
    label = str(label).lower().strip()

    mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "neg": "negative",
        "neu": "neutral",
        "pos": "positive",
    }

    return mapping.get(label, label)


def predict(text: str, model, tokenizer):
    """
    Predict one text example.

    Returns:
        predicted_label, predicted_probability
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_idx = int(torch.argmax(probabilities, dim=-1).item())
    predicted_probability = float(probabilities[0, predicted_idx].item())

    raw_label = model.config.id2label[predicted_idx]
    predicted_label = normalize_label(raw_label)

    return predicted_label, predicted_probability


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path.

    Returns a DataFrame with original columns plus:
    - predicted_label
    - predicted_probability
    - correct
    """
    df = pd.read_csv(adv_csv_path)

    rows = []

    for _, row in df.iterrows():
        predicted_label, predicted_probability = predict(row["text"], model, tokenizer)
        expected_label = normalize_label(row["expected_label"])

        result = row.to_dict()
        result["expected_label"] = expected_label
        result["predicted_label"] = predicted_label
        result["predicted_probability"] = predicted_probability
        result["correct"] = predicted_label == expected_label

        rows.append(result)

    return pd.DataFrame(rows)


def main() -> None:
    """
    Orchestrate adversarial evaluation and write results.csv.
    """
    model_path = os.getenv("MODEL_PATH", "model")
    adv_csv_path = os.getenv("ADVERSARIAL_SET_PATH", "adversarial_set.csv")
    output_path = os.getenv("OUTPUT_PATH", "results.csv")

    model, tokenizer = load_model(model_path)
    results = run_against_set(adv_csv_path, model, tokenizer)

    results.to_csv(output_path, index=False)

    print(f"Saved {output_path}")
    print("\nOverall accuracy:", round(results["correct"].mean(), 4))
    print("\nPer-category accuracy:")
    print(results.groupby("hypothesis_category")["correct"].mean().round(4))
    print("\nCorrect / incorrect counts:")
    print(
        results.groupby("hypothesis_category")["correct"]
        .value_counts()
        .unstack(fill_value=0)
    )


if __name__ == "__main__":
    main()
