# Week 3 — House Prices Baseline Regression Comparison

## Dataset and split

- Input file: `cleaned_ames_train.csv`
- Target: `SalePrice`
- Rows used: 1460
- Train rows: 1168 (80%)
- Test rows: 292 (20%)
- Random state: 42
- Dropped identifier columns, if present: Id
- scikit-learn version: 1.6.1

The raw feature data was split before any preprocessing was fitted. The imputer,
one-hot encoder, and scaler are contained in a `ColumnTransformer` inside each
model `Pipeline`; fitting each pipeline on `X_train` learns preprocessing
parameters from training rows only. Calling `predict(X_test)` applies those
already-fitted transformations to the test set without refitting them.
`OneHotEncoder(handle_unknown="ignore")` safely handles categories found only
in the test set.

## Test-set metrics

All metrics below are computed on the original `SalePrice` scale (currency units
as represented in the dataset). Lower MAE and RMSE are better; higher R² is better.

| Model                        |     MAE |    RMSE |      R2 |
|:-----------------------------|--------:|--------:|--------:|
| Mean baseline                | 62575.9 | 87619   | -0.0009 |
| Linear Regression            | 18388.3 | 29354.7 |  0.8877 |
| Ridge (alpha=0.1)            | 18342   | 28863.6 |  0.8914 |
| Ridge (alpha=1)              | 19047.6 | 29494.6 |  0.8866 |
| Ridge (alpha=10)             | 19153.2 | 30160.5 |  0.8814 |
| Ridge (alpha=100)            | 18868.1 | 31714.4 |  0.8689 |
| Log-target Linear Regression | 14808.3 | 22951.2 |  0.9313 |
| Log-target Ridge (alpha=0.1) | 15164.1 | 23331.3 |  0.929  |

## Interpretation

- **Mean baseline:** predicts the training-set mean SalePrice for every test row. It is a simple reference point, not a useful feature-based model.
- **Linear Regression vs. baseline:** Linear Regression has test MAE 18,388.28 versus baseline MAE 62,575.93, and RMSE 29,354.71 versus 87,619.03. Its RMSE reduction relative to baseline is 66.50%. R² is 0.8877 versus -0.0009.
- **Ridge vs. Linear Regression:** the best of the tested Ridge alpha values by test RMSE is Ridge (alpha=0.1), with RMSE 28,863.58. Compared with plain Linear Regression, its RMSE reduction is 1.67%. Treat a small difference cautiously: this is one fixed test split, not evidence of robust generalization by itself.
- **Log target experiment:** log-target Linear Regression RMSE is 22,951.18; log-target Ridge RMSE is 23,331.31. The predictions were inverse-transformed to the original SalePrice scale before scoring. A log transform can reduce the influence of expensive homes during fitting, but it is not guaranteed to improve original-scale RMSE or MAE.

**Conclusion:** Judge whether Ridge meaningfully improves over Linear Regression by the size and consistency of the metric changes, not by R² alone. Judge both against the mean baseline using MAE and RMSE as well as R². The results above are from this run's held-out test set; do not claim a model is better if the metric differences are tiny or inconsistent.

## Metric guide

- **MAE:** average absolute prediction error, in SalePrice units; relatively easy to interpret.
- **RMSE:** square root of mean squared error, in SalePrice units; penalizes large errors more strongly than MAE.
- **R²:** relative fit compared with predicting the test-set mean; 1 is perfect, 0 is no better than that reference, and it can be negative.

## Reproducibility

Run from the repository root:

```bash
python regression_baseline.py data/cleaned_house_prices.csv
```

Change the CSV path to match the cleaned Task 1 dataset. The run writes `MODEL_COMPARISON.md`
and `model_metrics.csv` to the dataset directory. Ridge alpha values can be changed with
`--alphas 0.1 1 10 100`. This task intentionally uses one 80/20 split; cross-validation
can be added in a later task to estimate performance variability.
