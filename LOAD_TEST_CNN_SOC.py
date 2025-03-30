import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

MODEL_PATH = "cnn_soc_model.pth"
SCALER_PATH = "cnn_soc_scaler.pkl"
NEW_DATA_PATH = "new_battery_data.csv"
OUTPUT_CSV = "new_soc_predictions.csv"
WINDOW_SIZE = 3
MC_ITER = 100

class Deep_CNN_MC(nn.Module):
    def __init__(self, in_channels=3, dropout_prob=0.1):
        super(Deep_CNN_MC, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

model = Deep_CNN_MC()
model.load_state_dict(torch.load(MODEL_PATH))
model.train()
scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(NEW_DATA_PATH)
X_raw = df[['Voltage_V', 'Current_A', 'Temperature_C']].values
time = df['Time_s'].values
true_soc = df['SOC'].values

def add_uncertainty(X, std_dev=[0.01, 0.05, 0.02]):
    return X + np.random.normal(0, std_dev, X.shape)

X_noisy = add_uncertainty(X_raw)

def create_sequences(X, time, window_size, y):
    X_seq, t_seq, y_seq = [], [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        t_seq.append(time[i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(t_seq), np.array(y_seq)

X_seq, time_seq, true_soc_seq = create_sequences(X_noisy, time, WINDOW_SIZE, y=true_soc)

X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
X_scaled = scaler.transform(X_seq_flat).reshape(X_seq.shape)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).permute(0, 2, 1)

def predict_with_uncertainty(model, x, n_iter=100):
    preds = [model(x).detach().numpy() for _ in range(n_iter)]
    preds = np.stack(preds).squeeze()
    return preds.mean(axis=0), preds.std(axis=0)

mean_pred, std_pred = predict_with_uncertainty(model, X_tensor, n_iter=MC_ITER)

results_df = pd.DataFrame({
    'Time_s': time_seq,
    'True_SOC': true_soc_seq,
    'Predicted_SOC': mean_pred,
    'Uncertainty_STD': std_pred
})

results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to: {OUTPUT_CSV}")

rms_error = np.sqrt(mean_squared_error(results_df['True_SOC'], results_df['Predicted_SOC']))
print(f"RMS Error: {rms_error:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(results_df['Time_s'], results_df['Predicted_SOC'], label='Predicted SOC')
plt.fill_between(results_df['Time_s'],
                 results_df['Predicted_SOC'] - results_df['Uncertainty_STD'],
                 results_df['Predicted_SOC'] + results_df['Uncertainty_STD'],
                 alpha=0.3, color='gray', label='Uncertainty')
plt.xlabel("Time (s)")
plt.ylabel("SOC")
plt.title("Predicted SOC with Uncertainty")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
