# Multimodal Anomaly Detection & Failure Prediction Project
# This project builds an end-to-end pipeline to detect anomalies and predict failures in large-scale systems using time-series, sensor, and log data.


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

# This code simulate multimodal data 

np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)

sensor_data = pd.DataFrame({
    "time": time,
    "temperature": 50 + 5*np.sin(time/50) + np.random.normal(0, 1, n_samples),
    "pressure": 30 + 3*np.cos(time/40) + np.random.normal(0, 1, n_samples),
    "vibration": np.random.normal(0, 1, n_samples)
})

# This code inject anomalies.
anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
sensor_data.loc[anomaly_indices, "vibration"] += 5

# This code indicate Failure labels.
sensor_data["failure"] = 0
sensor_data.loc[anomaly_indices, "failure"] = 1

# This code Log Data and do Embeddings.

print("Loading transformer model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_bert = AutoModel.from_pretrained("distilbert-base-uncased")

logs = ["system running normally"] * n_samples
for idx in anomaly_indices:
    logs[idx] = "error detected in subsystem vibration spike"

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

print("Generating log embeddings...")
log_embeddings = np.array([get_embedding(log) for log in logs])

# This code does Feature Fusion.

X_sensor = sensor_data[["temperature", "pressure", "vibration"]].values
X = np.hstack((X_sensor, log_embeddings))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = sensor_data["failure"].values

# This code does Anomaly Detection.

print("Running anomaly detection...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_preds = iso_forest.fit_predict(X_scaled)
anomaly_preds = (anomaly_preds == -1).astype(int)

sensor_data["anomaly_detected"] = anomaly_preds

# This code does LSTM Dataset.

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=10):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx+self.seq_len], dtype=torch.float32)
        )

dataset = TimeSeriesDataset(X_scaled, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# This code does LSTM Model.

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return self.sigmoid(out)

model = LSTMModel(input_size=X_scaled.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# This code does training on LSTM model. 

print("Training LSTM model...")
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# This code does evaluation of model.

print("Evaluating model...")
model.eval()
preds = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in loader:
        outputs = model(X_batch).squeeze()
        preds.extend((outputs > 0.5).int().numpy())
        actuals.extend(y_batch.numpy())

print(classification_report(actuals, preds))

# This code does visualization.

plt.figure(figsize=(12,5))
plt.plot(sensor_data["time"], sensor_data["vibration"], label="Vibration")
plt.scatter(sensor_data["time"], sensor_data["anomaly_detected"]*5,
            label="Detected Anomaly")
plt.legend()
plt.title("Anomaly Detection in Sensor Data")
plt.xlabel("Time")
plt.ylabel("Vibration")
plt.show()
