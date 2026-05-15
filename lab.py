"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def get_data_path() -> str:
    """
    Return DATA_PATH env var if set; otherwise return default dataset path.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(data_path: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    """
    Load the CSV at `data_path` and produce a train/test split.
    """
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    ds_dict = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )

    return ds_dict


def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.
    """
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return ds_dict.map(tokenize_fn, batched=True)


def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """
    Return TrainingArguments configured for fine-tuning.
    """
    try:
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            seed=seed,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            report_to="none",
            push_to_hub=False,
        )
    except TypeError:
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=lr,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            seed=seed,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            report_to="none",
            push_to_hub=False,
        )


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def write_training_log(trainer: Trainer) -> None:
    """
    Save Trainer log history so tests can confirm training happened.
    """
    with open("training_log.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Fine-tune a pre-trained model on the tokenized dataset.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    write_training_log(trainer)

    return trainer


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the fine-tuned classifier on the test split.
    """
    prediction_output = trainer.predict(tokenized_test)

    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    id2label = trainer.model.config.id2label
    label_ids = sorted(id2label.keys())

    per_class_f1_values = f1_score(
        labels,
        predictions,
        labels=label_ids,
        average=None,
        zero_division=0,
    )

    per_class_precision_values = precision_score(
        labels,
        predictions,
        labels=label_ids,
        average=None,
        zero_division=0,
    )

    per_class_recall_values = recall_score(
        labels,
        predictions,
        labels=label_ids,
        average=None,
        zero_division=0,
    )

    per_class_f1 = {
        id2label[label_id]: float(value)
        for label_id, value in zip(label_ids, per_class_f1_values)
    }

    per_class_precision = {
        id2label[label_id]: float(value)
        for label_id, value in zip(label_ids, per_class_precision_values)
    }

    per_class_recall = {
        id2label[label_id]: float(value)
        for label_id, value in zip(label_ids, per_class_recall_values)
    }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax over the last dimension.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)

    return exp / exp.sum(axis=-1, keepdims=True)


def write_predictions_csv(trainer: Trainer, tokenized_test, raw_test) -> pd.DataFrame:
    """
    Write predictions.csv with predicted label and probabilities.
    """
    prediction_output = trainer.predict(tokenized_test)

    logits = prediction_output.predictions
    pred_idx = np.argmax(logits, axis=1)
    pred_probs = _softmax(logits)

    id2label = trainer.model.config.id2label

    df_out = pd.DataFrame({
        "text": raw_test["text"],
        "label": [id2label[int(i)] for i in raw_test["label"]],
        "predicted_label": [id2label[int(i)] for i in pred_idx],
        "predicted_probability": [
            float(pred_probs[i, pred_idx[i]])
            for i in range(len(pred_idx))
        ],
    })

    for label_id in sorted(id2label.keys()):
        label_name = id2label[label_id]
        df_out[f"prob_{label_name}"] = [
            float(pred_probs[i, int(label_id)])
            for i in range(len(pred_idx))
        ]

    df_out.to_csv("predictions.csv", index=False)

    return df_out


def write_confusion_matrix_csv(trainer: Trainer, tokenized_test) -> pd.DataFrame:
    """
    Write confusion_matrix.csv as rows=true labels and columns=predicted labels.
    """
    prediction_output = trainer.predict(tokenized_test)

    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    predictions = np.argmax(logits, axis=1)

    id2label = trainer.model.config.id2label
    label_ids = sorted(id2label.keys())
    label_names = [id2label[i] for i in label_ids]

    cm = confusion_matrix(
        labels,
        predictions,
        labels=label_ids,
    )

    cm_df = pd.DataFrame(
        cm,
        index=label_names,
        columns=label_names,
    )

    cm_df.to_csv("confusion_matrix.csv")

    return cm_df


def write_evaluation_report(
    ds: DatasetDict,
    metrics: dict,
    cm_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    training_time_minutes: float,
) -> None:
    """
    Create evaluation-report.md.
    """
    train_size = len(ds["train"])
    test_size = len(ds["test"])
    total_size = train_size + test_size

    all_labels = [ID2LABEL[int(i)] for i in ds["train"]["label"]] + [
        ID2LABEL[int(i)] for i in ds["test"]["label"]
    ]

    label_distribution = pd.Series(all_labels).value_counts().to_dict()

    errors = predictions_df[
        predictions_df["label"] != predictions_df["predicted_label"]
    ].head(3)

    if len(errors) == 0:
        error_text = "- No misclassified examples were found in the test split.\n"
    else:
        error_lines = []

        for _, row in errors.iterrows():
            gold_label = row["label"]
            gold_prob = row.get(f"prob_{gold_label}", 0.0)

            error_lines.append(
                f"""- Original sentence: {row["text"]}
  - Gold label: {gold_label}
  - Predicted label: {row["predicted_label"]}
  - Predicted probability for the gold label: {gold_prob:.4f}
  - Why this may be wrong: This sentence may contain wording that overlaps with another sentiment class. The model likely focused on a misleading cue instead of the full meaning.
"""
            )

        error_text = "\n".join(error_lines)

    report = f"""# Module 7 Week A — Lab Evaluation Report

## Dataset

This lab uses the AARSynth app reviews sentiment dataset for three-class sentiment classification: negative, neutral, and positive. The dataset has {total_size} examples total, with {train_size} examples in the training split and {test_size} examples in the test split.

Label distribution: {label_distribution}

## Model and hyperparameters

- Backbone: distilbert-base-uncased
- Number of labels: 3
- Learning rate: 5e-5
- Epochs: 2
- Batch size: 8
- Max length: 128
- Seed: 42
- Training time: {training_time_minutes:.2f} minutes on my machine

## Metrics on the test split

Aggregate:

| Metric | Value |
|---|---:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Macro-F1 | {metrics["macro_f1"]:.4f} |

Per class:

| Class | F1 | Precision | Recall |
|---|---:|---:|---:|
| Negative | {metrics["per_class_f1"].get("negative", 0):.4f} | {metrics["per_class_precision"].get("negative", 0):.4f} | {metrics["per_class_recall"].get("negative", 0):.4f} |
| Neutral | {metrics["per_class_f1"].get("neutral", 0):.4f} | {metrics["per_class_precision"].get("neutral", 0):.4f} | {metrics["per_class_recall"].get("neutral", 0):.4f} |
| Positive | {metrics["per_class_f1"].get("positive", 0):.4f} | {metrics["per_class_precision"].get("positive", 0):.4f} | {metrics["per_class_recall"].get("positive", 0):.4f} |

## Confusion matrix

Rows are true labels and columns are predicted labels.

{cm_df.to_string()}

## Three qualitative error examples

{error_text}

## Hugging Face Hub model URL

https://huggingface.co/hurab3a/m7-app-review-sentiment
"""

    with open("evaluation-report.md", "w", encoding="utf-8") as f:
        f.write(report)


def main() -> None:
    """
    Orchestrate the full pipeline.
    """
    set_seed(42)

    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"

    start_time = time.time()

    ds = prepare_dataset(data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized = tokenize_dataset(ds, tokenizer, max_length=128)

    tokenized.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    training_args = make_training_args(
        output_dir=output_dir,
        lr=5e-5,
        epochs=2,
        batch_size=8,
        seed=42,
    )

    trainer = train_classifier(
        tokenized_ds=tokenized,
        model_name=model_name,
        training_args=training_args,
        tokenizer=tokenizer,
        num_labels=3,
    )

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_training_log(trainer)

    metrics = evaluate_classifier(trainer, tokenized["test"])

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions_df = write_predictions_csv(
        trainer=trainer,
        tokenized_test=tokenized["test"],
        raw_test=ds["test"],
    )

    cm_df = write_confusion_matrix_csv(
        trainer=trainer,
        tokenized_test=tokenized["test"],
    )

    training_time_minutes = (time.time() - start_time) / 60

    write_evaluation_report(
        ds=ds,
        metrics=metrics,
        cm_df=cm_df,
        predictions_df=predictions_df,
        training_time_minutes=training_time_minutes,
    )

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    print("\nConfusion matrix rows=true, cols=pred:")
    print(cm_df.to_string())

    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"

        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to https://huggingface.co/hurab3a/{repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            print("Run `huggingface-cli login` with a write token and try again.")


if __name__ == "__main__":
    main()