# Module 7 Week A — Lab Evaluation Report

## Dataset

This lab uses the AARSynth app reviews sentiment dataset for three-class sentiment classification: negative, neutral, and positive. The dataset has 7472 examples total, with 5977 examples in the training split and 1495 examples in the test split.

Label distribution: {'negative': 2519, 'positive': 2511, 'neutral': 2442}

## Model and hyperparameters

- Backbone: distilbert-base-uncased
- Number of labels: 3
- Learning rate: 5e-5
- Epochs: 2
- Batch size: 8
- Max length: 128
- Seed: 42
- Training time: 26.17 minutes on my machine

## Metrics on the test split

Aggregate:

| Metric | Value |
|---|---:|
| Accuracy | 0.6281 |
| Macro-F1 | 0.6263 |

Per class:

| Class | F1 | Precision | Recall |
|---|---:|---:|---:|
| Negative | 0.7060 | 0.7169 | 0.6954 |
| Neutral | 0.4774 | 0.4558 | 0.5011 |
| Positive | 0.6957 | 0.7171 | 0.6754 |

## Confusion matrix

Rows are true labels and columns are predicted labels.

|          |   negative |   neutral |   positive |
|:---------|-----------:|----------:|-----------:|
| negative |        347 |       134 |         18 |
| neutral  |        107 |       232 |        124 |
| positive |         30 |       143 |        360 |

## Three qualitative error examples

- Original sentence: good, but slow workflow.
  - Gold label: positive
  - Predicted label: neutral
  - Predicted probability for the gold label: 0.3391
  - Why this may be wrong: This sentence may contain wording that overlaps with another sentiment class. The model likely focused on a misleading cue instead of the full meaning.

- Original sentence: nice app to use with friends
  - Gold label: neutral
  - Predicted label: positive
  - Predicted probability for the gold label: 0.1154
  - Why this may be wrong: This sentence may contain wording that overlaps with another sentiment class. The model likely focused on a misleading cue instead of the full meaning.

- Original sentence: when i was being rained on it said no rain in your <url> it is still a good weather app.
  - Gold label: neutral
  - Predicted label: positive
  - Predicted probability for the gold label: 0.4587
  - Why this may be wrong: This sentence may contain wording that overlaps with another sentiment class. The model likely focused on a misleading cue instead of the full meaning.


## Hugging Face Hub model URL

https://huggingface.co/hurab3a/m7-app-review-sentiment
