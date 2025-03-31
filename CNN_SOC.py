import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('battery_sim_output.csv')
features = ['Voltage_V', 'Current_A', 'Temperature_C']
target = 'SOC'
X_raw = df[features].values
y_raw = df[target].values
time = df['Time_s'].values

def add_uncertainty(X, std_dev=[0.01, 0.05, 0.02]):
    return X + np.random.normal(0, std_dev, X.shape)

X_noisy = add_uncertainty(X_raw)

def create_sequences(X, y, time, window_size):
    X_seq, y_seq, t_seq = [], [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
        t_seq.append(time[i + window_size])
    return np.array(X_seq), np.array(y_seq), np.array(t_seq)

window_size = 3
X_seq, y_seq, t_seq = create_sequences(X_noisy, y_raw, time, window_size)

scaler = StandardScaler()
X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
X_seq_scaled = scaler.fit_transform(X_seq_flat).reshape(X_seq.shape)

X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
    X_seq_scaled, y_seq, t_seq, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

model.train()
def predict_with_uncertainty(model, x, n_iter=100):
    preds = [model(x).detach().numpy() for _ in range(n_iter)]
    preds = np.stack(preds).squeeze()
    return preds.mean(axis=0), preds.std(axis=0)

mean_pred, std_pred = predict_with_uncertainty(model, X_test_tensor)

results_df = pd.DataFrame({
    'Time_s': t_test,
    'True_SOC': y_test.flatten(),
    'Predicted_SOC': mean_pred,
    'Uncertainty_STD': std_pred
}).sort_values(by='Time_s')

rms_error = np.sqrt(mean_squared_error(results_df['True_SOC'], results_df['Predicted_SOC']))
print(f"RMS Error: {rms_error:.6f}")


torch.save(model.state_dict(), "cnn_soc_model.pth")
joblib.dump(scaler, "cnn_soc_scaler.pkl")
results_df.to_csv("cnn_soc_predictions.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(results_df['Time_s'], results_df['True_SOC'], label='True SOC', linewidth=2)
plt.plot(results_df['Time_s'], results_df['Predicted_SOC'], label='Predicted SOC (mean)', linewidth=2)
plt.fill_between(results_df['Time_s'],
                 results_df['Predicted_SOC'] - results_df['Uncertainty_STD'],
                 results_df['Predicted_SOC'] + results_df['Uncertainty_STD'],
                 alpha=0.3, color='gray', label='Uncertainty (std)')
plt.xlabel('Time (s)')
plt.ylabel('SOC')
plt.title('SOC Estimation with Deep CNN + MC Dropout')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


