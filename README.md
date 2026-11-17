# Time Series with Facebook Prophet

Google stock closing-price forecasting with **Prophet** and Bayesian hyperparameter tuning using **Optuna's Tree-structured Parzen Estimator (TPE)**.

## Dataset

`GOOGLE.csv` — 4,431 historical records from **2004-08-19 to 2022-03-24**. This project deliberately predicts the **100 NYSE trading sessions following 2022-03-24**, not the next 100 days from today's date. The model uses `Date` and the original `Close` (not `Adj Close`), and does not use test data during tuning. This source history predates later stock splits; be careful comparing the resulting historical price units with post-split prices.

## Getting started

Use Python 3.10–3.12 (recommended). Place `GOOGLE.csv` beside the Python script.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python "Google Stock Price Prediction.py" --data GOOGLE.csv --trials 20
```

For a faster initial test use `--trials 3 --tune-window 1000`. Full optimization can take significant time due to repeated Prophet fits.

## Method

1. Parse and clean dates/closing prices; sort chronologically; remove duplicate dates.
2. Separate the last 100 trading observations for a **strictly held-out test**, the 100 before them for **validation**, and earlier observations for **training**.
3. Bayesian optimization (Optuna TPE) minimizes validation RMSE over `changepoint_prior_scale`, `seasonality_prior_scale`, `seasonality_mode`, `changepoint_range`, `n_changepoints`, and yearly/weekly seasonality switches.
4. Refit tuned Prophet on train + validation; score test MAE, RMSE and MAPE against a last-close (constant) baseline.
5. Refit tuned Prophet on the **full historical dataset** and forecast the next 100 **NYSE market sessions**; save 95% model-based uncertainty intervals.

**Note:** A single validation block can still overfit during optimization; multiple rolling-origin folds would be a stronger extension. Prophet trend extrapolation may be unsuitable for volatile stock prices; compare results with the naive baseline before claiming improvements. The displayed Prophet intervals are model assumptions, not calibrated investment-risk bounds.

## Output files

The `outputs/` directory contains `forecast_100_days.csv`, `holdout_predictions.csv`, `optimization_trials.csv`, `metrics.json`, `forecast.png`, and `test_predictions.png`.

## GitHub publishing

```bash
git init
git add .
git commit -m "Add Prophet Google stock forecast with Bayesian optimization"
git branch -M main
gh auth login
gh repo create "Time Series with Facebook Prophet" --public --source=. --remote=origin --push
```

GitHub repo *names* can contain spaces in some UI contexts, but GitHub CLI may normalize the actual URL slug. If the CLI rejects the name, use `time-series-with-facebook-prophet` as the repository slug and **Time Series with Facebook Prophet** as the repository's display name/description.

## Open-source contribution checklist

Find actively maintained repositories with a clear contributing guide. Fork, clone, make a scoped change, test it, commit, push your feature branch, then open a PR explaining the problem, solution and tests. PRs must contain genuine fixes or features; submitting empty or cosmetic-only PRs solely to satisfy an assignment is discouraged.

```bash
gh repo fork OWNER/REPO --clone
git checkout -b fix/meaningful-improvement
# edit, test, commit
git add . && git commit -m "Fix: explain the change"
git push -u origin fix/meaningful-improvement
gh pr create --repo OWNER/REPO --base main --head YOURUSERNAME:fix/meaningful-improvement
```

## Reproducibility

`--seed 42` fixes Optuna's sampler randomness. Fitting can still differ across Prophet/Stan versions and platforms. The CSV is included for a self-contained assignment deliverable.
