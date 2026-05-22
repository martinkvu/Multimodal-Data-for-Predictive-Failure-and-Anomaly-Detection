# Multimodal-Data-for-Predictive-Failure-and-Anomaly-Detection

# 🔍 Multimodal Anomaly Detection & Failure Prediction

> An end-to-end ML pipeline that predicts hardware failures from IoT sensor signals using time-series modeling, transformer log embeddings, and AWS cloud integration.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20CloudWatch-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | **97%** |
| Failure Precision | **0.89** |
| Failure Recall | **0.80** |
| Failure F1 Score | **0.84** |
| Macro Avg F1 | **0.91** |
| Anomalies Detected | **50/50 (100%)** |
| False Positives | **0** |

---

## 🧠 How It Works

The pipeline runs **8 steps** end-to-end:

1. **Data Ingestion** — sensor readings (temperature, pressure, vibration) loaded from S3 or generated locally
2. **Log Embeddings** — system logs encoded via `DistilBERT` into 768-dim vectors
3. **Feature Fusion** — sensor features + log embeddings concatenated and standardized
4. **Anomaly Detection** — `Isolation Forest` flags anomalies in an unsupervised pass
5. **LSTM Training** — sequence model learns failure patterns across 10-step windows
6. **Evaluation** — precision, recall, F1, and accuracy computed and printed
7. **Cloud Publishing** — metrics pushed to CloudWatch; model artifact saved to S3
8. **Visualization** — 4-panel dashboard saved as `anomaly_detection_results.png`

---

## ⚙️ Requirements

**Python 3.8+** is required.

Install all dependencies with:

```bash
pip3 install boto3 torch transformers scikit-learn pandas numpy matplotlib
```

| Package | Purpose |
|---------|---------|
| `torch` | LSTM model training |
| `transformers` | DistilBERT log embeddings |
| `scikit-learn` | Isolation Forest, scaler, metrics |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` | Visualization dashboard |
| `boto3` | AWS S3 and CloudWatch integration |

---

## 🚀 How to Run

### Local mode *(no AWS credentials needed)*

```bash
python3 Multimodal_Failure_Detection.py
```

> Cloud calls are mocked by default. You will see `[MOCK S3]` and `[MOCK CloudWatch]` in the logs. The model trains fully and results are printed and saved locally.

### Cloud mode *(with AWS credentials)*

```bash
export USE_CLOUD=true
python3 Multimodal_Failure_Detection.py
```

---

## ☁️ AWS Setup

### 1. Configure credentials

```bash
aws configure
```

Enter your `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and region (`us-east-1` by default).

### 2. Create an S3 bucket

```bash
aws s3 mb s3://anomaly-detection-pipeline
```

The pipeline writes to the following structure automatically:

```
s3://anomaly-detection-pipeline/
├── sensor-data/{job_id}/sensor_data.csv
├── models/lstm_{job_id}.pt
└── results/{job_id}/results.json
```

### 3. IAM permissions required

```json
{
  "Effect": "Allow",
  "Action": [
    "cloudwatch:PutMetricData",
    "s3:PutObject",
    "s3:GetObject"
  ],
  "Resource": "*"
}
```

> Metrics publish under the namespace `AnomalyDetection/Pipeline`, tagged by `JobId`.

---

## 🔁 SageMaker (Optional)

The training loop is **SageMaker-compatible**. To run as a managed job, point a SageMaker `PyTorch` estimator at this script as the entry point and pass `USE_CLOUD=true` via the `environment` config.

---

## 📁 Project Structure

```
Multimodal_Failure_Detection.py   # Main pipeline script
anomaly_detection_results.png     # Generated visualization (after run)
README.md                         # This file
```

---

## 🛠️ Configuration

Edit these constants at the top of the script to customize the pipeline:

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `S3_BUCKET` | `anomaly-detection-pipeline` | S3 bucket name |
| `CW_NAMESPACE` | `AnomalyDetection/Pipeline` | CloudWatch namespace |
| `USE_CLOUD` | `false` | Set to `true` to enable real AWS calls |
| `n_samples` | `1000` | Number of sensor timesteps |
| `epochs` | `5` | LSTM training epochs |
