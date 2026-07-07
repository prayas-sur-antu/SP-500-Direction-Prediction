# S&P 500 Market Direction Prediction (Random Forest, Backtesting)

A machine-learning project that predicts whether the S&P 500 index will close higher the next day, using a Random Forest classifier evaluated with a proper time-series backtest. This started as a Dataquest portfolio project; I worked through it to understand time-series validation and feature engineering for financial data.

Honest framing up front: this is a methods exercise in backtesting and feature engineering, not a trading strategy. Predicting daily market direction is close to a coin flip, and the results here reflect that.

## What it does

- Downloads the full S&P 500 daily price history with `yfinance`.
- Defines a binary target: whether tomorrow's close is higher than today's.
- Trains a Random Forest classifier and — importantly — evaluates it with a **backtest** that trains on the past and predicts forward in steps, so the model is never trained on future data (no lookahead bias).
- Engineers features across multiple time horizons (2, 5, 60, 250, 1000 days): close-to-rolling-average ratios and recent up-day trend counts.
- Uses a probability threshold (0.6) so the model only predicts "up" when it is more confident, trading coverage for higher precision.

## Results

Precision is the metric of interest here — of the days the model predicts "up," how often it is right — because a prediction is only useful if it's better than the base rate of up-days (~54%).

| Stage | Precision | Base rate (up-days) |
|-------|-----------|--------------------|
| Simple model, single test split | ~0.47 | ~0.54 |
| Backtested baseline model | ~0.53 | ~0.54 |
| + engineered features + 0.6 threshold | ~0.57 | ~0.55 |

The first model does worse than simply assuming the market goes up. Proper backtesting brings it in line with the base rate, and the engineered features plus a stricter threshold produce a small edge (~57% vs a ~55% base rate). That gap is marginal and would not survive trading costs — which is the realistic and expected outcome. The point of the project is the process, not the profit.

## What this project is really about

The transferable skill here is **evaluating a time-series model honestly**: you can't shuffle time-ordered data or use a random train/test split without leaking future information, so the backtest trains only on the past at each step. This same discipline applies to any sequential or temporal prediction problem.

## Tech stack

Python, pandas, scikit-learn (RandomForestClassifier), yfinance.
