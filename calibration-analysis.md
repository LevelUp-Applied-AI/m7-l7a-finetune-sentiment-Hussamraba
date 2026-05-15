# Module 7 Week A — Stretch Calibration Analysis

## Reliability diagram interpretation

The reliability diagram compares the model's confidence with its actual accuracy in each confidence bucket. A perfectly calibrated model would follow the diagonal line, where confidence and accuracy are equal.

In my results, the model is not perfectly calibrated. The low-confidence buckets from 0.05 to 0.25 had no examples, so they do not provide useful evidence. The 0.35 bucket had only 5 examples and showed 0.8000 accuracy, so I would not rely too much on that bucket because the sample size is very small.

The most important pattern appears in the mid-confidence buckets. The 0.55 confidence bucket had accuracy of 0.4094 with 254 examples, and the 0.65 confidence bucket had accuracy of 0.5181 with 276 examples. This means the model was often over-confident in this range because its confidence was higher than its real accuracy.

The higher-confidence buckets performed better. The 0.85 bucket had 0.7179 accuracy with 319 examples, and the 0.95 bucket had 0.8721 accuracy with 305 examples. These buckets are closer to the ideal calibration line, but the model is still not perfectly calibrated.

## Expected Calibration Error

The Expected Calibration Error for this model is:

**ECE = 0.1194**

This means that, on average, the model's confidence differs from its actual accuracy by about 11.94 percentage points across the confidence buckets.

For production use, this means the classifier's probability scores should not be trusted as perfect probabilities. The model can still be useful for sentiment classification, but a prediction with 0.80 confidence does not always mean the model is correct 80% of the time.

## A specific calibration pattern

A specific calibration pattern I observed is over-confidence in the middle confidence range. For example, the 0.55 bucket had only 0.4094 accuracy, and the 0.65 bucket had only 0.5181 accuracy.

This likely happened because the model was trained mainly to improve classification accuracy using cross-entropy loss, not to produce perfectly calibrated probabilities. Also, sentiment labels can be difficult when app reviews contain mixed opinions, short phrases, or neutral language that looks similar to positive or negative language.

The neutral class was also the weakest class in the manual evaluation. It had lower precision, recall, and F1 compared with negative and positive. This may explain why the model becomes over-confident when it sees examples near the boundary between neutral and the other sentiment classes.

## A proposed engineering action

In production, I would use threshold-based abstention. For example, if the model's maximum predicted probability is below 0.70, the system should avoid making an automatic decision and send the review to a human reviewer or mark it as uncertain.

I would also consider temperature scaling on a validation set. Temperature scaling can make the predicted probabilities better calibrated without changing the model's predicted class. This would make the confidence scores more trustworthy for production decisions.

A third action would be to collect more difficult neutral examples and mixed-sentiment reviews. Since the neutral class is harder for the model, adding more training data around these boundary cases could improve both classification quality and calibration.