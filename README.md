# Bitcoin Multi-Horizon Price Prediction with Continuous Training

An MLOps project for forecasting Bitcoin (BTC) prices across multiple time horizons, built with a continuous training pipeline that adapts to data drift and performance degradation in the highly dynamic cryptocurrency market.

**Course:** Machine Learning Operations (MLOps)  
**Instructor:** Rizal Setya Perdana, S.Kom., M.Kom., Ph.D.  
**Author:** Erza Hanif Pramudita Hanggara (245150200111038)  
**Institution:** Faculty of Computer Science, Universitas Brawijaya, 2026

---

## Overview

Cryptocurrency markets trade around the clock and can shift dramatically in short periods of time. A model trained once on historical data will inevitably degrade as market conditions evolve. This project addresses that problem by building a prediction system that:

- Forecasts BTC prices at **multiple horizons** (1h, 3h, 6h, 12h, 24h) using a **single regression model**, with horizon treated as an input feature rather than requiring separate models per horizon.
- Continuously ingests new market data.
- Monitors for **data drift** and **performance degradation**.
- Automatically triggers **retraining** when needed, registering new model versions for deployment.

## Problem Domain

- **Domain:** Digital finance / cryptocurrency (Bitcoin)
- **ML Task:** Regression for time-series forecasting
- **Approach:** Multi-horizon forecasting via a single model, using horizon as an input feature

```
Market data up to time t → horizon → model → predicted price at t + horizon
```

## Data

**Source:** [CoinGecko API](https://www.coingecko.com/en/api)

Historical market data is pulled at an hourly resolution and includes:

| Field | Description |
|---|---|
| `timestamp` | Unix epoch timestamp (ms) |
| `prices` | BTC price at time *t* |
| `market_caps` | Market capitalization (price × circulating supply) |
| `total_volumes` | Total trading volume at time *t* |

Example response structure:

```json
{
  "prices": [[timestamp, price]],
  "market_caps": [[timestamp, market_cap]],
  "total_volumes": [[timestamp, volume]]
}
```

## Project Structure
```
project
├── data
│   ├── raw/      
│   └── processed/      
├── models/
├── src/
├── notebooks/  
├── tests/  
├── docs/  
├── configs/               
└── README.md
```

## Running with Codespace
1. Open this repository on GitHub.  
2. Click **Code → Codespaces → Create codespace on main**.  
3. Wait for the environment to be automatically set up using `.devcontainer/devcontainer.json`.  
4. Run the following command in the Codespaces terminal:
   ```bash
   python src/hello.py```