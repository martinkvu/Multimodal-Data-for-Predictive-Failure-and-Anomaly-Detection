# Multimodal-Data-for-Predictive-Failure-and-Anomaly-Detection


Multimodal Anomaly Detection & Failure Prediction
An end-to-end ML pipeline that detects sensor-based system failures using time-series sensor data and log embeddings, deployed with AWS cloud integration.

Results
MetricScoreOverall Accuracy97%Failure Precision0.89Failure Recall0.80Failure F1 Score0.84Macro Avg F10.91Anomalies Detected50/50 (100%)False Positives0

How It Works
The pipeline has 8 steps that run end-to-end:

Data Ingestion — sensor readings (temperature, pressure, vibration) are loaded from S3 or generated locally
Log Embeddings — system log messages are encoded using DistilBERT (768-dim embeddings)
Feature Fusion — sensor features and log embeddings are concatenated and standardized
Anomaly Detection — Isolation Forest flags anomalous readings in an unsupervised pass
LSTM Training — a sequence model learns failure patterns across 10-step windows
Evaluation — precision, recall, F1, and accuracy are computed and printed
Cloud Publishing — metrics pushed to CloudWatch; model artifact saved to S3
Visualization — 4-panel matplotlib dashboard saved as anomaly_detection_results.png


Requirements
Python version
Python 3.8 or higher
Install dependencies
bashpip3 install boto3 torch transformers scikit-learn pandas numpy matplotlib
Full dependency list
PackagePurposetorchLSTM model trainingtransformersDistilBERT log embeddingsscikit-learnIsolation Forest, StandardScaler, metricspandas / numpyData manipulationmatplotlibVisualization dashboardboto3AWS S3 and CloudWatch integrationbotocoreAWS error handling (installed with boto3)

How to Run
Local mode (no AWS credentials needed)
bashpython3 Multimodal_Failure_Detection.py
Cloud calls are mocked by default — you will see [MOCK S3] and [MOCK CloudWatch] log lines. The model trains fully and results are printed and saved locally.
Cloud mode (with AWS credentials)
bashexport USE_CLOUD=true
python3 Multimodal_Failure_Detection.py
This activates real S3 uploads and CloudWatch metric publishing. See AWS setup below.

AWS Setup
To use cloud mode you need three things:
1. AWS credentials configured
bashaws configure
Enter your AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and preferred region (us-east-1 by default).
2. S3 bucket created
Create a bucket named anomaly-detection-pipeline (or update S3_BUCKET in the script):
bashaws s3 mb s3://anomaly-detection-pipeline
The pipeline writes to three prefixes automatically:
s3://anomaly-detection-pipeline/
├── sensor-data/{job_id}/sensor_data.csv
├── models/lstm_{job_id}.pt
└── results/{job_id}/results.json
3. CloudWatch permissions
Your IAM user or role needs the following permissions:
json{
  "Effect": "Allow",
  "Action": [
    "cloudwatch:PutMetricData",
    "s3:PutObject",
    "s3:GetObject"
  ],
  "Resource": "*"
}
Metrics are published under the namespace AnomalyDetection/Pipeline and tagged by JobId.

SageMaker (Optional)
The training loop is structured to be SageMaker-compatible. To run as a managed training job, replace the local training block with a SageMaker PyTorch estimator pointed at this script as the entry point. The USE_CLOUD=true environment variable can be passed via environment in the estimator config.

Output
After a successful run you will have:

Terminal — printed results table with all citable metrics
anomaly_detection_results.png — 4-panel dashboard (vibration signal, loss curve, anomaly counts, classifier metrics)
S3 (cloud mode) — sensor CSV, model .pt file, results JSON
CloudWatch (cloud mode) — per-epoch loss and final evaluation metrics


Project Structure
Multimodal_Failure_Detection.py   # Main pipeline script
anomaly_detection_results.png     # Generated visualization (after run)
README.md                         # This file

Configuration
Key constants at the top of the script you may want to change:
VariableDefaultDescriptionAWS_REGIONus-east-1AWS regionS3_BUCKETanomaly-detection-pipelineS3 bucket nameCW_NAMESPACEAnomalyDetection/PipelineCloudWatch namespaceUSE_CLOUDfalseSet to true to enable real AWS callsn_samples1000Number of sensor timestepsepochs5LSTM training epochs
