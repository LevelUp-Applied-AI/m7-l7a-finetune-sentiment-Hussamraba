# Adversarial Evaluation: App-Review Sentiment Classifier

## Per-hypothesis accuracy

This adversarial evaluation tested the Lab 7A fine-tuned sentiment classifier on 30 hand-crafted examples. Each example was designed to target a specific hypothesized failure mode rather than to represent a random held-out test sample.

| Hypothesis category | Total | Correct | Incorrect | Accuracy |
|---|---:|---:|---:|---:|
| negation | 5 | 3 | 2 | 0.6000 |
| lexical_trigger | 5 | 2 | 3 | 0.4000 |
| domain_shift | 5 | 2 | 3 | 0.4000 |
| length_extreme | 5 | 3 | 2 | 0.6000 |
| sarcasm | 5 | 1 | 4 | 0.2000 |
| other | 5 | 2 | 3 | 0.4000 |
| overall | 30 | 13 | 17 | 0.4333 |

The overall accuracy on the adversarial set was 0.4333. This is much lower than I would expect on a normal held-out app-review test set, but that is the point of this exercise. The adversarial set was designed to stress the model’s weak points, especially negation, surface cue words, sarcasm, and domain-shifted factual language.

## Confirmed hypotheses

The clearest confirmed weakness was sarcasm. The model only got 1 out of 5 sarcasm examples correct. Row 21 was expected to be negative, but the model predicted positive with probability 0.8582 for the sentence: “Fantastic, another update that makes the app crash faster.” This strongly confirms that the model reads the positive surface cue “Fantastic” more literally than the intended sarcastic meaning.

Rows 22, 24, and 25 show the same pattern. Row 22 was expected to be negative, but the model predicted positive with probability 0.9136. Row 24 was expected to be negative, but the model predicted positive with probability 0.8936. Row 25 was expected to be negative, but the model predicted positive with probability 0.8027. These errors suggest that the model does not reliably understand sarcastic complaints when the sentence starts with positive words like “Amazing,” “Perfect,” or “Great.”

The lexical-trigger hypothesis was also confirmed. Row 6 was expected to be negative, but the model predicted neutral with probability 0.6427 for the sentence: “The interface looks beautiful, but the app deletes my saved work.” The model did not fully treat the second clause as the main sentiment. Row 9 showed a similar problem: it was expected to be negative, but the model predicted neutral with probability 0.5708, even though the inability to complete a payment is a serious negative issue.

Domain shift was another confirmed weakness. Row 12 was expected to be neutral because it is a factual sports sentence, but the model predicted negative with probability 0.8319. Row 14 was also expected to be neutral, but the model predicted negative with probability 0.7582. These results suggest that the model treats negative event words like “lost” and “criticized” as sentiment signals, even when the sentence is just reporting news.

## Refuted hypotheses

Some examples were handled better than expected. In the negation category, the model correctly predicted row 1 as negative with probability 0.7574: “The app is not useful anymore after the last update.” It also correctly predicted row 3 as negative with probability 0.7747: “The update did not improve performance at all.” I expected negation to be a larger weakness, but these rows show that the model can sometimes handle simple negation when the sentence still looks like a normal app review.

The model also handled some length extremes better than expected. Row 16, “Works fine,” was correctly predicted as positive with probability 0.8658. Row 18, “Okay, nothing special,” was correctly predicted as neutral with probability 0.8206. Row 19, a long mixed review, was correctly predicted as neutral with probability 0.6940. This suggests the model is not simply failing because of very short or very long text. It can still classify length extremes when the wording contains familiar review-style signals.

The domain-shift category was not a total failure either. Row 13, a recipe instruction, was correctly predicted as neutral with probability 0.7393. Row 15, an entertainment news sentence about a movie opening at number one, was also correctly predicted as neutral with probability 0.4403. These examples show that the model can sometimes recognize factual non-review text as neutral, especially when the sentence does not contain strong negative or positive event cues.

## What the results reveal about the decision boundary

The adversarial results suggest that the model’s decision boundary is strongly influenced by surface-level sentiment cues and familiar app-review patterns. Words like “Fantastic,” “Amazing,” “Perfect,” “Great,” “lost,” “criticized,” “broken,” and “crashes” appear to push the model toward positive or negative labels, even when the full sentence meaning is more complicated.

The model seems more reliable when the sentence resembles a normal app review. It handled simple examples like “Works fine,” “Okay, nothing special,” and “The update did not improve performance at all.” However, it struggled when the meaning depended on sarcasm, contrast, or factual reporting. This suggests that the model has learned useful review-language associations, but it does not consistently understand speaker intent or the difference between factual negative events and negative sentiment.

The biggest decision-boundary insight is that the classifier is not only separating sentiment by meaning. It is also separating by cue words and familiar review patterns. This works well for many normal app reviews, but it becomes risky when the same cue words appear in sarcastic, domain-shifted, or contrast-heavy examples. In production, I would not trust this model on adversarial or out-of-domain text without additional targeted evaluation and fine-tuning.
