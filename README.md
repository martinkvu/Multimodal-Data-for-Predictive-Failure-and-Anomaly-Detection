# Multimodal-Data-for-Predictive-Failure-and-Anomaly-Detection




An end-to-end ML pipeline that detects sensor-based system failures using time-series sensor data and log embeddings, deployed with AWS cloud integration.

Results: The model achieved 97% overall accuracy with a failure class precision of 0.89, recall of 0.80, and F1 score of 0.84. The Isolation Forest detected all 50 injected anomalies (100%) with zero false positives.

How it works: The pipeline runs 8 steps end-to-end. Sensor readings (temperature, pressure, vibration) are ingested from S3 or generated locally, then system log messages are encoded using DistilBERT into 768-dimensional embeddings. Those embeddings are fused with the sensor features, standardized, and passed through an Isolation Forest for unsupervised anomaly detection. An LSTM then learns failure patterns across 10-step windows, after which precision, recall, F1, and accuracy are computed. Finally, metrics are pushed to CloudWatch, the model artifact is saved to S3, and a 4-panel visualization dashboard is saved locally.

Requirements: Python 3.8 or higher. Install all dependencies with pip3 install boto3 torch transformers scikit-learn pandas numpy matplotlib. The main packages are PyTorch for the LSTM, HuggingFace Transformers for DistilBERT, scikit-learn for Isolation Forest and metrics, and boto3 for AWS integration.

How to run: By default the pipeline runs in local mode with no AWS credentials needed — just run python3 Multimodal_Failure_Detection.py. Cloud calls are mocked and results are printed and saved locally. To activate real AWS integration, set USE_CLOUD=true before running.

AWS setup: You need three things to use cloud mode. First, configure your AWS credentials using aws configure with your access key, secret key, and region. Second, create an S3 bucket named anomaly-detection-pipeline — the pipeline automatically writes sensor CSVs, model .pt files, and a results JSON to prefixed folders inside it. Third, make sure your IAM user has cloudwatch:PutMetricData, s3:PutObject, and s3:GetObject permissions. Metrics publish under the namespace AnomalyDetection/Pipeline tagged by job ID.

SageMaker: The training loop is structured to be SageMaker-compatible. Swap the local training block for a SageMaker PyTorch estimator using this script as the entry point and pass USE_CLOUD=true via the environment config.

Key settings you can change: AWS region (default us-east-1), S3 bucket name, CloudWatch namespace, number of sensor timesteps (default 1,000), and number of training epochs (default 5).
