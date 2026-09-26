# Bitcoin Multi-Horizon Price Prediction with Continuous Training

An MLOps project for forecasting Bitcoin (BTC) prices across multiple time horizons, built with a continuous training pipeline that adapts to data drift and performance degradation in the highly dynamic cryptocurrency market.

**Author:** Erza Hanif Pramudita Hanggara  

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
The raw response from the API is stored in JSON format in data/raw/. Each ingestion creates a new file with a timestamp in its filename so that previously raw data is not overwritten

Example:
```
data/raw/ 
├── bitcoin_data_20260925_100000.json 
├── bitcoin_data_20260925_110000.json 
└── bitcoin_data_20260925_120000.json
```

## Data Pipeline
The current data pipeline follows this flow:
```
CoinGecko API --> Data Ingestion --> Raw JSON --> Data Preprocessing --> Processed CSV
```

The complete pipeline can be executed using:

```python src/run_pipeline.py```

The pipeline performs two main stages:
1. **Data ingestion** using ```python src/ingest_data.py```
2. **Data preprocessing** using ```python src/preprocess.py```

## Data Ingestion

It uses the CoinGecko API to retrieve Bitcoin market data and stores the response as a timestamped JSON file in: ```data/raw/```

Run the ingestion script manually with:

```python src/ingest_data.py```

Example output:
```
Fetching Bitcoin data from CoinGecko...
Data saved to: ../data/raw bitcoin_data_20260925_120000.json
Data ingestion completed.
```

The timestamped filename allows the script to be executed repeatedly without destructively overwriting previously collected raw data.

## Data Preprocessing
The preprocessing stage:
- Loads raw JSON files from data/raw/.
- Converts the JSON data into a tabular DataFrame.
- Converts Unix timestamps from milliseconds into - UTC datetime.
- Sorts records chronologically.
- Removes duplicate timestamps.
- Removes the currently incomplete hourly record.
- Combines the processed data with the existing - historical dataset.
- Saves the resulting dataset to: ```data/processed/bitcoin.csv```

Run preprocessing manually with:

```python src/preprocess.py```

The current hourly record is excluded because the hour is still in progress and therefore does not yet represent a complete observation. The record will be ingested again after the hour has finished. For example, if ingestion occurs at 14:14, the latest available data point is 14:14. Since this is not aligned to the full hour (00 minutes), it is removed.

## Project Structure
```
project
├── .devcontainer/ 
│   ├── devcontainer.json 
│   └── Dockerfile 
│
├── .github/ 
│   └── workflows/
│   └── bitcoin_pipeline.yml 
│
├── data
│   ├── raw/      
│   └── processed/
│         └── bitcoin_data.csv
│  
├── models/
├── src/
│   ├── ingest_data.py 
│   ├── preprocess.py 
│   └── run_pipeline.py
│
├── notebooks/  
├── tests/  
├── docs/  
├── configs/
├── requirements.txt               
└── README.md
```

## Running with Codespace
1. Open this repository on GitHub.  
2. Click **Code → Codespaces → Create codespace on main**.  
3. Wait for the environment to be automatically set up using `.devcontainer/devcontainer.json`.  
4. The required Python dependencies are installed automatically.
5. Run the complete data pipeline:

    ```python src/run_pipeline.py```

    To run only data ingestion:
    ```python src/ingest_data.py```

    To run only preprocessing:
    ```python src/preprocess.py```

## Automated Periodic Pipeline
The data pipeline is automated using GitHub Actions.

The workflow configuration is located at:

```.github/workflows/bitcoin_pipeline.yml```

The workflow can be triggered manually through GitHub Actions using workflow_dispatch and is also configured to run periodically using a scheduled cron trigger, which is every hour.

This allows the data ingestion and preprocessing process to be executed periodically without manually running the scripts.

**Limitation** : As currently there are no DVC yet, the scheduler won't automically save the process and immediately remove it from memory, therefore to save the file, wrokflow need to continuously commit and push and this was not practical for a github repository.