#!/usr/bin/env python3
"""
Week 3: Leakage-safe House Prices regression baseline.

Usage:
    python regression_baseline.py path/to/cleaned_house_prices.csv
    python regression_baseline.py path/to/cleaned_house_prices.csv --target SalePrice

Expected input: the cleaned Task 1 Ames/House Prices dataset, with SalePrice
as a column. The script writes MODEL_COMPARISON.md and model_metrics.csv beside
the input file (or to --output-dir if provided).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build transformers from training features only; fit happens inside Pipeline."""
    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to the cleaned Task 1 CSV")
    parser.add_argument("--target", default="SalePrice", help="Target column (default: SalePrice)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0],
                        help="Ridge alpha values to compare (default: 0.1 1 10 100)")
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {args.csv_path}\n"
            "Pass the path to your cleaned Task 1 dataset, e.g. data/cleaned_house_prices.csv."
        )
    if len(args.alphas) < 3:
        raise ValueError("Provide at least three Ridge alpha values.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    df = pd.read_csv(args.csv_path)
    if args.target not in df.columns:
        raise ValueError(
            f"Target column {args.target!r} not found. Columns include: {list(df.columns[:20])}"
        )
    df = df.loc[df[args.target].notna()].copy()
    y = pd.to_numeric(df[args.target], errors="raise")
    X = df.drop(columns=[args.target])

    # Common Ames dataset files include an identifier column, not a predictive feature.
    # Drop it if present; adjust this list if Task 1 established a different policy.
    id_columns = [c for c in X.columns if c.lower() in {"id", "parcelid", "parcel_id"}]
    if id_columns:
        X = X.drop(columns=id_columns)
        print(f"Dropped identifier column(s): {id_columns}")

    # IMPORTANT: split raw features before fitting any imputer, encoder, or scaler.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Infer numeric/categorical columns using training data only.
    preprocessor = build_preprocessor(X_train)

    # Mean-prediction baseline does not need feature preprocessing.
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    rows = [{"Model": "Mean baseline", **regression_metrics(y_test, baseline.predict(X_test))}]

    linear_pipeline = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            ("model", LinearRegression()),
        ]
    )
    linear_pipeline.fit(X_train, y_train)
    rows.append({"Model": "Linear Regression", **regression_metrics(y_test, linear_pipeline.predict(X_test))})

    ridge_rows = []
    for alpha in args.alphas:
        pipeline = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", Ridge(alpha=alpha)),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = regression_metrics(y_test, pipeline.predict(X_test))
        row = {"Model": f"Ridge (alpha={alpha:g})", **metrics}
        rows.append(row)
        ridge_rows.append((alpha, pipeline, metrics))

    # Test whether log1p(SalePrice) helps. Predictions are inverse-transformed back
    # to dollars before metrics are calculated, so metrics remain comparable.
    # TransformedTargetRegressor fits the target transform only through training y.
    best_alpha, _, _ = min(ridge_rows, key=lambda item: item[2]["RMSE"])
    log_linear = TransformedTargetRegressor(
        regressor=clone(linear_pipeline),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=True,
    )
    log_linear.fit(X_train, y_train)
    rows.append({
        "Model": "Log-target Linear Regression",
        **regression_metrics(y_test, log_linear.predict(X_test)),
    })

    log_ridge = TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocess", clone(preprocessor)),
            ("model", Ridge(alpha=best_alpha)),
        ]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=True,
    )
    log_ridge.fit(X_train, y_train)
    rows.append({
        "Model": f"Log-target Ridge (alpha={best_alpha:g})",
        **regression_metrics(y_test, log_ridge.predict(X_test)),
    })

    results = pd.DataFrame(rows)
    results["MAE"] = results["MAE"].round(2)
    results["RMSE"] = results["RMSE"].round(2)
    results["R2"] = results["R2"].round(4)

    output_dir = args.output_dir or args.csv_path.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "model_metrics.csv"
    report_path = output_dir / "MODEL_COMPARISON.md"
    results.to_csv(metrics_path, index=False)

    baseline_row = results.loc[results["Model"] == "Mean baseline"].iloc[0]
    linear_row = results.loc[results["Model"] == "Linear Regression"].iloc[0]
    best_ridge_row = results[results["Model"].str.startswith("Ridge (")].sort_values("RMSE").iloc[0]
    log_linear_row = results.loc[results["Model"] == "Log-target Linear Regression"].iloc[0]
    log_ridge_row = results[results["Model"].str.startswith("Log-target Ridge")].iloc[0]

    def pct_reduction(candidate: float, baseline_value: float) -> float:
        return (baseline_value - candidate) / baseline_value * 100 if baseline_value else float("nan")

    report = f"""# Week 3 — House Prices Baseline Regression Comparison

## Dataset and split

- Input file: `{args.csv_path.name}`
- Target: `{args.target}`
- Rows used: {len(df)}
- Train rows: {len(X_train)} ({(1-args.test_size)*100:.0f}%)
- Test rows: {len(X_test)} ({args.test_size*100:.0f}%)
- Random state: {args.random_state}
- Dropped identifier columns, if present: {', '.join(id_columns) if id_columns else 'none'}
- scikit-learn version: {sklearn.__version__}

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

{results.to_markdown(index=False)}

## Interpretation

- **Mean baseline:** predicts the training-set mean SalePrice for every test row. It is a simple reference point, not a useful feature-based model.
- **Linear Regression vs. baseline:** Linear Regression has test MAE {linear_row['MAE']:,.2f} versus baseline MAE {baseline_row['MAE']:,.2f}, and RMSE {linear_row['RMSE']:,.2f} versus {baseline_row['RMSE']:,.2f}. Its RMSE reduction relative to baseline is {pct_reduction(float(linear_row['RMSE']), float(baseline_row['RMSE'])):.2f}%. R² is {linear_row['R2']:.4f} versus {baseline_row['R2']:.4f}.
- **Ridge vs. Linear Regression:** the best of the tested Ridge alpha values by test RMSE is {best_ridge_row['Model']}, with RMSE {best_ridge_row['RMSE']:,.2f}. Compared with plain Linear Regression, its RMSE reduction is {pct_reduction(float(best_ridge_row['RMSE']), float(linear_row['RMSE'])):.2f}%. Treat a small difference cautiously: this is one fixed test split, not evidence of robust generalization by itself.
- **Log target experiment:** log-target Linear Regression RMSE is {log_linear_row['RMSE']:,.2f}; log-target Ridge RMSE is {log_ridge_row['RMSE']:,.2f}. The predictions were inverse-transformed to the original SalePrice scale before scoring. A log transform can reduce the influence of expensive homes during fitting, but it is not guaranteed to improve original-scale RMSE or MAE.

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
"""
    report_path.write_text(report, encoding="utf-8")

    print("\nTest-set model comparison:")
    print(results.to_string(index=False))
    print(f"\nWrote: {report_path}")
    print(f"Wrote: {metrics_path}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=UserWarning)
        main()
