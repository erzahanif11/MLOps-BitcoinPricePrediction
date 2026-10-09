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

The current latest record is excluded because it is still in progress and therefore does not yet represent a complete observation. For example, if ingestion occurs at 14:14, the latest available data point is 14:14. Since this is not aligned to the full hour (00 minutes), it is removed, leaving only the data point at 14.00.

## Data Versioning with DVC
DVC (Data Version Control) is used to track dataset versions independently of Git's regular file tracking. This allows changes to the dataset to be recorded without committing the full dataset contents directly to Git.

DVC metadata is stored in .dvc files, which reference the corresponding data objects. These metadata files can be committed to Git so that dataset versions can be associated with specific repository revisions

### Track Dataset Changes
After modifying the processed dataset, check its DVC status:

`dvc status`

To compare the current data state with a previous version:

`dvc diff`

These commands help identify whether tracked data has changed and which dataset changes are associated with a version transition.

### Store and Retrieve Versioned Data

After configuring the DVC remote, upload tracked data to remote storage:

`dvc push`

Retrieve the tracked data from remote storage:

`dvc pull`

### Why Data Versioning Matters

Data versioning is essential for ensuring the reproducibility of machine learning experiments. In time-series forecasting, newly ingested Bitcoin market data can change the training dataset and consequently affect model performance.

Without dataset versioning, it may be difficult to reproduce previous experiments or determine whether a change in model performance was caused by changes in the model, its configuration, or the training data.

DVC tracks dataset versions through metadata files, while Git records the corresponding metadata and code history. MinIO stores the actual versioned data objects remotely.

This separation enables the project to:

- **Reproduce experiments:** Retrieve the dataset associated with a specific Git revision.
- **Track data changes:** Identify when the training dataset changes between experiments.
- **Maintain data integrity:** Preserve previous data versions instead of relying solely on the latest processed CSV.
- **Support model comparison:** Evaluate different model versions using identifiable dataset versions and consistent experimental configurations.

For reproducible results, the dataset version, code revision, preprocessing configuration, and model parameters should be recorded together for each experiment.

## Remote Storage with MinIO

MinIO is used as the S3-compatible object storage backend for DVC and is hosted in the local environment.

Instead of storing the actual dataset objects directly in the Git repository, DVC uploads them to the configured MinIO bucket. Git tracks the corresponding DVC metadata, allowing specific dataset versions to be identified and retrieved when needed.

The data versioning workflow is:

```text
Processed Dataset
       |
       v
  DVC Tracking
       |
       +----> DVC Metadata (Git)
       |
       v
    DVC Push
       |
       v
MinIO Object Storage
```

The remote configuration specifies the MinIO endpoint and bucket. Access credentials must be configured locally or through environment variables and must not be committed to the repository.

To inspect the configured DVC remotes, run:

```bash
dvc remote list
```

### Cloudflare Tunnel Configuration

This project uses a **Cloudflare Quick Tunnel** to expose the locally hosted MinIO S3 API endpoint to the development environment.

Because a Quick Tunnel generates a temporary URL, the endpoint may change whenever the tunnel is restarted. When this happens, update the DVC remote endpoint configuration to use the newly generated URL before performing remote operations such as `dvc push` or `dvc pull`.

Update the endpoint using:

```bash
dvc remote modify minio_remote endpointurl https://<new-tunnel-url>
```

Replace `minio_remote` with the configured DVC remote name and `<new-tunnel-url>` with the current tunnel URL.

After updating the endpoint, verify the remote configuration and ensure that the tunnel is running before synchronizing data with MinIO.

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

## DVC and MinIO Setup

This section describes how to configure DVC with MinIO as an S3-compatible remote storage backend. A tunnel is used to expose the MinIO endpoint when it is not directly accessible from the development environment.

### 1. Install DVC with S3 Support

Install DVC with its S3 storage dependencies:

```bash
pip install "dvc[s3]"
```

Verify the installation:

```bash
dvc --version
```

### 2. Prepare MinIO Object Storage

Ensure that a MinIO instance is running and accessible. Create a bucket to store DVC-managed data objects.

Record the following configuration details:

- MinIO endpoint URL
- Bucket name
- Access key
- Secret key

The endpoint must be reachable from the environment running DVC. Do not commit access credentials or other secrets to the Git repository.

### 3. Expose MinIO Through a Tunnel

If MinIO is not directly accessible from the development environment, establish a tunnel to expose its S3 API endpoint.

Configure the tunnel to forward traffic to the MinIO API port on the machine or environment where MinIO is running. Use the resulting reachable endpoint as the DVC remote endpoint.

The tunnel must remain active whenever DVC needs to communicate with MinIO. Use the actual endpoint and tunnel command from your environment; do not assume that a locally accessible MinIO address is also accessible from GitHub Codespaces.

### 4. Configure the DVC Remote

Configure a named DVC remote using the MinIO endpoint and bucket:

```bash
dvc remote add -d minio_remote s3://<bucket-name>/<path>
dvc remote modify minio_remote endpointurl <endpoint-url>
```

Replace the placeholders with the actual bucket, optional object prefix, and reachable endpoint.

Configure the access credentials using environment variables or another supported secure credential mechanism. Avoid placing secrets directly in tracked configuration files.

Verify the remote configuration:

```bash
dvc remote list
```

### 5. Track the Dataset

Track the processed dataset with DVC:

```bash
dvc add data/processed/bitcoin_data.csv
dvc add data/raw/
```

This creates DVC metadata for the dataset and updates `.gitignore` so that the data file is not tracked directly by Git.

Commit the DVC metadata and relevant repository changes:

```bash
git add data/processed/bitcoin_data.csv.dvc .gitignore
git commit -m "Track processed dataset with DVC"
```

### 6. Upload Data to MinIO

Upload the tracked dataset to the configured remote:

```bash
dvc push
```

Check the remote synchronization status:

```bash
dvc status -c
```

### 7. Verify Dataset Versioning

Modify or regenerate the dataset, then check its status:

```bash
dvc status
```

After updating the DVC-tracked dataset, inspect the changes:

```bash
dvc add data/processed/bitcoin_data.csv
dvc diff
```

Commit the updated DVC metadata and upload the new data version:

```bash
git add data/processed/bitcoin_data.csv.dvc .gitignore
git commit -m "Update processed dataset version"
dvc push
```

The updated metadata records the new dataset version, while Git history allows the previous metadata to be recovered.

### 8. Restore a Previous Dataset Version

Check out the Git revision containing the desired DVC metadata:

```bash
git checkout <commit-hash>
dvc pull
```

This restores the dataset corresponding to that revision, provided the required data objects remain available in MinIO and the remote is accessible.

### Notes

- Git tracks the source code and DVC metadata; MinIO stores the remote data objects.
- DVC does not automatically upload every dataset modification. The relevant tracking and `dvc push` commands must be executed.
- Remote operations require valid credentials and an active connection to the MinIO endpoint.
- A tunnel endpoint may change between sessions, depending on the tunneling service and its configuration.

## Automated Periodic Pipeline
The data pipeline is automated using GitHub Actions.

The workflow configuration is located at:

```.github/workflows/bitcoin_pipeline.yml```

The workflow can be triggered manually through GitHub Actions using workflow_dispatch and is also configured to run periodically using a scheduled cron trigger, which is every hour.

This allows the data ingestion and preprocessing process to be executed periodically without manually running the scripts.

## Project Structure
```
worspace/
 ├──.dvc
 │   └── config
 └── project
    ├── .devcontainer/ 
    │   ├── devcontainer.json 
    │   └── Dockerfile 
    │
    ├── .github/ 
    │   └── workflows/
    │   └── bitcoin_pipeline.yml 
    │
    ├── data (managed through dvc)
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
