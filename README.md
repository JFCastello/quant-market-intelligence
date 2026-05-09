# Quant Market Intelligence

Daily market analytics platform for regime detection, volatility forecasting, benchmark-vs-ML comparison, and structural break visualization.

## Overview

Quant Market Intelligence is a local-first analytics product for daily market data. Its first operational question is deliberately narrow and testable:

**Does a machine learning model add consistent out-of-sample value over a serious classical benchmark for 5-day volatility forecasting?**

The system is built around a reproducible end-to-end pipeline that:

- downloads daily OHLCV data for a small liquid universe,
- normalizes and validates raw and cleaned data,
- builds versioned features and targets,
- trains a classical volatility benchmark and an ML regressor,
- evaluates both with walk-forward temporal validation,
- detects structural breaks for interpretability,
- and exposes the results in a local Streamlit dashboard.

## Product question

The project is not designed around “using ML because it is fashionable.”  
Its central product question is:

> Can ML outperform a serious classical benchmark in a way that is consistent, out-of-sample, and strong enough to justify promotion in a real product setting?

## Current Phase 1 scope

Phase 1 is a **local MVP**, but it is **not** a throwaway prototype.  
It uses the same domain contracts, features, targets, benchmark family, ML family, evaluation logic, and service layer that are intended to survive into Phase 2. Phase 2 adds serving, storage adapters, API, jobs, containers, and deployment concerns, without redefining the analytical core.

### Core analytical scope

- **Frequency:** daily (`1d`)
- **Universe:** `SPY`, `TLT`, `GLD`, `HYG`
- **Main continuous target:** `future_rv_5d`
- **Secondary discrete target:** `future_regime_5d`
- **Benchmark:** `GARCH(1,1)` with Student-t innovations
- **ML model:** `XGBoost Regressor`
- **Structural break detection:** `ruptures` / PELT
- **Local interface:** Streamlit

## What the system does today

The current local Phase 1 system can:

- ingest and persist raw + normalized daily market data,
- build base and context features,
- build the forward target `future_rv_5d`,
- materialize walk-forward `train / validation / test` splits,
- generate regime targets by split using `train_only` thresholds,
- train and score a GARCH benchmark,
- map benchmark forecasts to discrete regimes,
- train and score an XGBoost regressor,
- compare benchmark vs ML with continuous and discrete metrics,
- apply a formal promotion / no-promotion decision rule,
- detect structural breaks,
- expose the results through a Streamlit dashboard backed by a Python service layer. 

## Main targets

### `future_rv_5d`

The main target is **5-day forward realized volatility**.

At date `t`, `future_rv_5d` is defined from returns observed in the future window:

- `t+1`
- `t+2`
- `t+3`
- `t+4`
- `t+5`

This is a **forward-looking** target, not a backward rolling volatility. It is designed so both the benchmark and the ML model compete on the same forecasting objective. 

### `future_regime_5d`

The discrete regime target is derived from `future_rv_5d` by applying quantile thresholds computed on the training split only, producing classes such as:

- `calm`
- `normal`
- `stress` 

## Evaluation philosophy

The project uses **walk-forward temporal validation** with explicit `train`, `validation`, and `test` windows. Thresholds for the regime target are computed from `train` only to avoid leakage. The system compares benchmark and ML using both continuous and discrete metrics.

### Main continuous metrics

- `QLIKE`
- `RMSE`
- `MAE`

### Main discrete metrics

- `Macro-F1`
- `Balanced Accuracy`

### Calibration note

The current v1 pipeline works with hard regime labels, but does **not** yet materialize validated probability outputs for regime classes. Because of that, probabilistic calibration is currently marked as **not evaluable yet** in the decision layer.

## Model promotion rule

ML is **not** promoted just because it looks interesting in one chart.

The current decision logic requires ML to:

- improve average out-of-sample `QLIKE`,
- avoid degrading regime classification quality,
- satisfy explicit promotion constraints before replacing the benchmark in product logic.

In the current v1 state, the benchmark remains the default recommendation across the active universe. 

## Local dashboard

The Streamlit dashboard is the local interface for Phase 1.  
Its role is to make the system understandable for an analyst without reimplementing business logic in the UI.

The current dashboard focuses on four main views:

- **Home / Executive Summary**
- **Market & Forecast**
- **Model Comparison**
- **Structural Changes**

The UI is designed to sit on top of a reusable Python service layer, not directly on top of raw scripts.

## Repository philosophy
This repository is designed with **continuity between Phase 1 and Phase 2** in mind.
That means:
- same data provider,
- same domain contracts,
- same features and targets,
- same benchmark and ML family,
- same evaluation logic,
- same service layer philosophy,
- later adding only storage adapters, API, jobs, containers, and deployment concerns. 

## Repository structure

```text
quant-market-intelligence/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── configs/
│   ├── base.yaml
│   ├── local.yaml
│   └── prod.yaml
├── data/
│   ├── raw/
│   ├── normalized/
│   ├── features/
│   ├── features_context/
│   └── targets/
├── artifacts/
│   ├── models/
│   ├── evaluations/
│   └── reports/
├── docs/
│   ├── architecture/
│   └── runbooks/
├── scripts/
├── streamlit_app/
│   ├── Home.py
│   └── pages/
├── src/quant_platform/
│   ├── providers/
│   ├── storage/
│   ├── schemas/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── services/
│   └── utils/
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
