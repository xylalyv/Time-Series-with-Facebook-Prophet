#!/usr/bin/env python3
"""Google closing-price forecasting with Prophet and Bayesian optimization (Optuna).

Example:
  python 'Google Stock Price Prediction.py' --data GOOGLE.csv --trials 20

Forecast horizon: 100 NYSE trading sessions following the dataset's final date.
This is a historical demonstration when the supplied CSV is dated 2022.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import exchange_calendars as xcals
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

LOGGER = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Load the original close (not adjacent columns) using Prophet's ds/y format."""
    raw = pd.read_csv(path)
    required = {'Date', 'Close'}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    frame = raw.loc[:, ['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    frame['ds'] = pd.to_datetime(frame['ds'], errors='coerce')
    frame['y'] = pd.to_numeric(frame['y'], errors='coerce')
    frame = frame.dropna(subset=['ds', 'y']).sort_values('ds')
    if frame['ds'].duplicated().any():
        LOGGER.warning('Duplicate dates found: retaining the last row per date')
        frame = frame.drop_duplicates(subset='ds', keep='last')
    frame = frame.reset_index(drop=True)
    if len(frame) < 400:
        raise ValueError('At least 400 clean historical observations are required')
    if (frame['y'] <= 0).any():
        raise ValueError('Closing prices must be positive for this forecasting task')
    return frame


def build_model(params: dict, uncertainty_samples: int = 0) -> Prophet:
    return Prophet(
        growth='linear',
        changepoint_prior_scale=float(params['changepoint_prior_scale']),
        seasonality_prior_scale=float(params['seasonality_prior_scale']),
        seasonality_mode=params['seasonality_mode'],
        changepoint_range=float(params['changepoint_range']),
        n_changepoints=int(params['n_changepoints']),
        yearly_seasonality=bool(params['yearly_seasonality']),
        weekly_seasonality=bool(params['weekly_seasonality']),
        daily_seasonality=False,
        interval_width=0.95,
        uncertainty_samples=uncertainty_samples,
    )


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        'MAE': float(mean_absolute_error(actual, predicted)),
        'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
        'MAPE_percent': float(np.mean(np.abs((actual - predicted) / actual)) * 100),
    }


def predict_on(model: Prophet, dates: pd.Series) -> np.ndarray:
    return model.predict(pd.DataFrame({'ds': pd.to_datetime(dates).to_numpy()}))['yhat'].to_numpy()


def tuning_objective(trial: optuna.Trial, train: pd.DataFrame, validation: pd.DataFrame) -> float:
    params = {
        'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
        'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True),
        'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
        'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95),
        'n_changepoints': trial.suggest_int('n_changepoints', 10, 35),
        'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
        'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False]),
    }
    model = build_model(params)
    model.fit(train)
    pred = predict_on(model, validation['ds'])
    return float(np.sqrt(mean_squared_error(validation['y'], pred)))


def nyse_future_dates(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    """Get actual NYSE market sessions, excluding weekends and exchange holidays."""
    cal = xcals.get_calendar('XNYS')
    start = pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1)
    # A 3-calendar-year search comfortably covers up to 500 requested sessions.
    end = start + pd.Timedelta(days=max(3 * periods, 366))
    available = cal.sessions_in_range(start, end).tz_localize(None)
    if len(available) < periods:
        raise ValueError('Exchange calendar does not provide enough future sessions')
    return available[:periods]


def save_plots(df, validation, test, test_pred, future_result, out_dir):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df['ds'], df['y'], label='Historical closing price', linewidth=1.25)
    ax.plot(future_result['ds'], future_result['yhat'], label='Prophet 100-session forecast', linewidth=2)
    ax.fill_between(future_result['ds'], future_result['yhat_lower'], future_result['yhat_upper'], alpha=0.2,
                    label='95% Prophet uncertainty interval')
    ax.axvline(df['ds'].iloc[-1], linestyle='--', alpha=0.6)
    ax.set(title='Google closing price: history and 100 NYSE-session Prophet forecast',
           xlabel='Date', ylabel='Historical close (CSV price units)')
    ax.legend(); ax.grid(alpha=0.25); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out_dir / 'forecast.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test['ds'], test['y'], label='Actual test close')
    ax.plot(test['ds'], test_pred, label='Prophet out-of-sample prediction')
    ax.set(title='Chronological holdout test: actual vs predicted', xlabel='Date', ylabel='Close')
    ax.legend(); ax.grid(alpha=0.25); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out_dir / 'test_predictions.png', dpi=160); plt.close(fig)


def run(args):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = load_data(args.data)
    if args.test_size < 20 or args.validation_size < 20:
        raise ValueError('Validation and test sizes must be at least 20 sessions each')
    if len(df) <= args.validation_size + args.test_size + 200:
        raise ValueError('Too little training data after holdouts')
    args.output.mkdir(parents=True, exist_ok=True)

    train_end = len(df) - args.validation_size - args.test_size
    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:-args.test_size].copy()
    trainval = df.iloc[:-args.test_size].copy()
    test = df.iloc[-args.test_size:].copy()

    LOGGER.info('Records: %s | range: %s to %s', len(df), df.ds.min().date(), df.ds.max().date())
    LOGGER.info('Train: %d | validation: %d | test: %d', len(train), len(validation), len(test))
    # Limit the expensive optimization-fit training history only when explicitly requested.
    tuning_train = train.tail(args.tune_window) if args.tune_window else train
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(lambda t: tuning_objective(t, tuning_train, validation), n_trials=args.trials,
                   show_progress_bar=False)
    best = study.best_params
    LOGGER.info('Best validation RMSE: %.3f', study.best_value)

    # Evaluate once on untouched test observations, without using test feedback to tune.
    holdout_model = build_model(best)
    holdout_model.fit(trainval)
    test_pred = predict_on(holdout_model, test['ds'])
    test_metrics = score(test['y'].to_numpy(), test_pred)
    naive = np.full(len(test), trainval['y'].iloc[-1])
    naive_metrics = score(test['y'].to_numpy(), naive)
    LOGGER.info('Prophet holdout: %s', test_metrics)
    LOGGER.info('Last-close baseline holdout: %s', naive_metrics)

    # Fit again on all CSV rows only after completing the unbiased test.
    final_model = build_model(best, uncertainty_samples=1000)
    final_model.fit(df)
    dates = nyse_future_dates(df['ds'].iloc[-1], args.horizon)
    raw_forecast = final_model.predict(pd.DataFrame({'ds': dates}))
    forecast = raw_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast = forecast.rename(columns={
        'yhat': 'predicted_close', 'yhat_lower': 'lower_95', 'yhat_upper': 'upper_95'
    })
    forecast.to_csv(args.output / 'forecast_100_days.csv', index=False)
    pd.DataFrame({'ds': test['ds'], 'actual_close': test['y'], 'predicted_close': test_pred}).to_csv(
        args.output / 'holdout_predictions.csv', index=False)
    pd.DataFrame([{'trial': t.number, 'rmse': t.value, **t.params}
                  for t in study.trials]).to_csv(args.output / 'optimization_trials.csv', index=False)
    report = {
        'dataset': str(args.data), 'rows': len(df),
        'historical_start': str(df.ds.min().date()), 'historical_end': str(df.ds.max().date()),
        'validation_rows': len(validation), 'test_rows': len(test),
        'best_validation_rmse': study.best_value, 'best_hyperparameters': best,
        'test_prophet': test_metrics, 'test_last_close_baseline': naive_metrics,
        'forecast_sessions': len(forecast), 'forecast_start': str(forecast.ds.iloc[0].date()),
        'forecast_end': str(forecast.ds.iloc[-1].date()),
        'interpretation': 'Historical demonstration using original dataset Close; not a live 2026 forecast.',
        'warning': 'Unadjusted historical closes may be incomparable across later stock splits; forecasts are not investment advice.'
    }
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save_plots(df, validation, test, test_pred,
               raw_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], args.output)
    print(json.dumps(report, indent=2))
    print(f'\nFiles saved in: {args.output.resolve()}')


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('GOOGLE.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--horizon', type=int, default=100, help='Future NYSE trading sessions')
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna TPE optimization trials')
    parser.add_argument('--validation-size', type=int, default=100)
    parser.add_argument('--test-size', type=int, default=100)
    parser.add_argument('--tune-window', type=int, default=None,
                        help='Optional last N training rows for faster optimization; final model uses all rows')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.trials < 1 or args.horizon < 1 or args.horizon > 500:
        parser.error('--trials >= 1 and 1 <= --horizon <= 500 are required')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    run(args)


if __name__ == '__main__':
    cli()
