# XGBoost Tuning & Model Comparison Notes

## XGBoost Hyperparameter Tuning

GridSearchCV was run over n_estimators [300, 500], max_depth [6, 10],
learning_rate [0.05, 0.1] with 3-fold stratified cross-validation
(8 candidates, 24 fits, 522 seconds). The best configuration found was
n_estimators=500, max_depth=10, learning_rate=0.1 -- identical to the
project's existing default hyperparameters, confirming they were already
optimal within the tested search space.

With 5-fold CV on the default config, XGBoost achieved 96.42% +/- 0.07%
accuracy on the training data.

## RF vs XGBoost Comparison

Both models were trained on the full preprocessed dataset (104,882 training
rows, 24 PCA-reduced features, 6 traffic classes) using stratified 5-fold
cross-validation before a final fit on the complete training set, then
evaluated on the held-out test set.

| Metric              | XGBoost | Random Forest |
|----------------------|---------|----------------|
| Accuracy             | 0.9641  | 0.9584         |
| Precision (macro)    | 0.9443  | 0.9326         |
| Recall (macro)       | 0.9360  | 0.9321         |
| F1 (macro)           | 0.9398  | 0.9322         |
| F1 (weighted)        | 0.9640  | 0.9586         |
| ROC-AUC (OvR)        | 0.9973  | 0.9970         |

XGBoost outperformed Random Forest on every evaluation metric. The margin
is modest but consistent, suggesting XGBoost's sequential boosting corrects
errors on the harder minority classes (Live Streaming, Buffered Video,
Music) slightly better than Random Forest's bagging approach.

## Conclusion

XGBoost is the stronger classifier for this task and is the recommended
model for the final system, with Random Forest serving as a solid,
slightly-faster-to-train baseline.