import numpy as np
import pandas as pd
import torch
import re
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator

import joblib
import matplotlib.pyplot as plt
import subprocess

# Run Parameters
lookup_table_path = "battery_lookup_table.csv"
simulation_output_path = "battery_sim_output.csv"
test_data_path = "new_battery_data.csv"

cnn_predictions_output_path = "cnn_soc_predictions.csv"
ekf_predictions_output_path = "ekf_soc_predictions.csv"

model_output_path = "cnn_soc_model.pth"
scaler_output_path = "cnn_soc_scaler.pkl"

generate_simulation_data = False
generate_test_data = False
to_train_cnn_model = False



# Hysterisis Simulation Parmaters
k0 = 0.005   
k1 = 0.0002    
k2 = 0.01    

# Cell Initial Parameters
capacity_Ah = 14.0
dt = 1.0
V_min_cutoff = 2.9
V_max_cutoff = 4.4
soc = 0.9
initial_T_ambient = 25.0

# CNN Parameters
window_size = 3
MC_ITER = 100

# Gamma and M Optimization Parameters
learning_rate = 0.01
max_iters = 100
tolerance = 1e-4
gamma = 1.14
M = 0.445

# EKF Constants
Qcell = 50400  # As
eta_tilda = 0.98 # tuned Coulombic efficiency


class ECMWithHysteresis:
    def __init__(self, R0=0.0035, R1=0.007, C1=6761,
                 R2=0.0001, C2=1206,
                 dt=1.0, mass=0.4, cp=1500, tau=1200,
                 V_min=3.0, V_max=4.35, T_ambient=25):
        self.R0, self.R1, self.C1 = R0, R1, C1
        self.R2, self.C2 = R2, C2
        self.dt = dt
        self.mass = mass
        self.cp = cp
        self.tau = tau
        self.T = T_ambient
        self.T_ambient = T_ambient
        self.V_min = V_min
        self.V_max = V_max
        self.V_rc1 = 0.0
        self.V_rc2 = 0.0
        self.hys_state = 0.0

    def voc_from_soc(self, soc):
        soc = soc * 100
        return (1.443e-10 * soc**5 - 2.747e-8 * soc**4 + 3.083e-6 * soc**3
            - 2.883e-4 * soc**2 + 0.02454 * soc + 2.995)

    def step(self, I, soc):
        OCV = self.voc_from_soc(soc)

        alpha_hys = 0.01
        self.hys_state += alpha_hys * (np.sign(I) - self.hys_state)
        hys_gain = k0 + k1 * abs(I) + k2 * (1 - soc)**2
        V_oc_hys = OCV + hys_gain * self.hys_state

        alpha1 = np.exp(-self.dt / (self.R1 * self.C1))
        alpha2 = np.exp(-self.dt / (self.R2 * self.C2))
        self.V_rc1 = alpha1 * self.V_rc1 + (1 - alpha1) * I * self.R1
        self.V_rc2 = alpha2 * self.V_rc2 + (1 - alpha2) * I * self.R2

        V_drop = I * self.R0 + self.V_rc1 + self.V_rc2
        V_terminal = V_oc_hys - V_drop

        Q = I**2 * self.R0 + I * (OCV - V_terminal)
        dT = (Q / (self.mass * self.cp)) - (self.T - self.T_ambient) / self.tau
        self.T += dT * self.dt

        return V_terminal, self.T
    
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

def logistic(x):
    return 1 / (1 + np.exp(-x))

def inverse_logistic(y):
    return np.log(y / (1 - y))

def add_uncertainty(X, std_dev=[0.01, 0.05, 0.02]):
    return X + np.random.normal(0, std_dev, X.shape)

def create_sequences_train(X, y, time, window_size):
    X_seq, y_seq, t_seq = [], [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
        t_seq.append(time[i + window_size])
    return np.array(X_seq), np.array(y_seq), np.array(t_seq)

def create_sequences_test(X, y, time, window_size):
    X_seq, t_seq, y_seq = [], [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        t_seq.append(time[i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(t_seq), np.array(y_seq)

def predict_with_uncertainty(model, x, n_iter=100):
    preds = [model(x).detach().numpy() for _ in range(n_iter)]
    preds = np.stack(preds).squeeze()
    return preds.mean(axis=0), preds.std(axis=0)

def optimize_gamma_m():
    x_gamma = inverse_logistic(gamma / 2)  # since gamma is around 1
    x_M = inverse_logistic(M)

    # === Optimization loop ===
    rmse_prev = None

    for i in range(max_iters):
        gamma = 2 * logistic(x_gamma)
        M = logistic(x_M)

        print(f"Iteration {i+1}: gamma = {gamma:.5f}, M = {M:.5f}")

        # Call the EKF script and extract RMSE
        result = subprocess.run(["python3", "MECH501_EKF_A2.py", str(gamma), str(M)],
                                stdout=subprocess.PIPE, text=True)

        match = re.search(r"RMSE between EKF and True SOC: ([0-9.]+)", result.stdout)
        if not match:
            print("EKF script failed or RMSE not found.")
            break

        rmse = float(match.group(1))
        print(f"RMSE: {rmse:.6f}")

        if rmse_prev is not None and abs(rmse_prev - rmse) < tolerance:
            print("Converged.")
            break

        # Finite difference gradient estimation
        eps = 1e-4

        # Gradient w.r.t x_gamma
        gamma_eps = 2 * logistic(x_gamma + eps)
        result_eps = subprocess.run(["python3", "MECH501_EKF_A2.py", str(gamma_eps), str(M)],
                                    stdout=subprocess.PIPE, text=True)
        match_eps = re.search(r"RMSE between EKF and True SOC: ([0-9.]+)", result_eps.stdout)
        if not match_eps:
            break
        rmse_gamma = float(match_eps.group(1))
        grad_gamma = (rmse_gamma - rmse) / eps

        # Gradient w.r.t x_M
        M_eps = logistic(x_M + eps)
        result_eps = subprocess.run(["python3", "MECH501_EKF_A2.py", str(gamma), str(M_eps)],
                                    stdout=subprocess.PIPE, text=True)
        match_eps = re.search(r"RMSE between EKF and True SOC: ([0-9.]+)", result_eps.stdout)
        if not match_eps:
            break
        rmse_M = float(match_eps.group(1))
        grad_M = (rmse_M - rmse) / eps

        # Update parameters
        x_gamma -= learning_rate * grad_gamma
        x_M -= learning_rate * grad_M
        rmse_prev = rmse

    print(f"Final gamma = {2 * logistic(x_gamma):.5f}, M = {logistic(x_M):.5f}, RMSE = {rmse:.6f}")


def load_cnn_model_from_file(model_path, scaler_path):
    model = Deep_CNN_MC()
    model.load_state_dict(torch.load(model_path))
    model.train()
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict_with_cnn(data_path, model, scaler, output_path, display_results=False):
    df = pd.read_csv(data_path)
    X_raw = df[['Voltage_V', 'Current_A', 'Temperature_C']].values
    time = df['Time_s'].values
    true_soc = df['SOC'].values

    X_noisy = add_uncertainty(X_raw)

    X_seq, time_seq, true_soc_seq = create_sequences_test(X_noisy, y=true_soc, time=time, window_size=window_size)

    X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_scaled = scaler.transform(X_seq_flat).reshape(X_seq.shape)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).permute(0, 2, 1)

    mean_pred, std_pred = predict_with_uncertainty(model, X_tensor, n_iter=MC_ITER)

    results_df = pd.DataFrame({
        'Time_s': time_seq,
        'True_SOC': true_soc_seq,
        'Predicted_SOC': mean_pred,
        'Uncertainty_STD': std_pred
    })

    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    rms_error = np.sqrt(mean_squared_error(results_df['True_SOC'], results_df['Predicted_SOC']))
    print(f"CNN RMS Error: {rms_error:.6f}")

    if display_results:
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
    
    return results_df, rms_error


def train_cnn_model(input_path, model_output_path, scaler_output_path, display_results=False, predictions_output_path=""):
    df = pd.read_csv(input_path)
    features = ['Voltage_V', 'Current_A', 'Temperature_C']
    target = 'SOC'
    X_raw = df[features].values
    y_raw = df[target].values
    time = df['Time_s'].values

    X_noisy = add_uncertainty(X_raw)
    X_seq, y_seq, t_seq = create_sequences_train(X_noisy, y_raw, time, window_size)

    scaler = StandardScaler()
    X_seq_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_seq_scaled = scaler.fit_transform(X_seq_flat).reshape(X_seq.shape)

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X_seq_scaled, y_seq, t_seq, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

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

    torch.save(model.state_dict(), model_output_path)
    joblib.dump(scaler, scaler_output_path)

    mean_pred, std_pred = predict_with_uncertainty(model, X_test_tensor)

    results_df = pd.DataFrame({
        'Time_s': t_test,
        'True_SOC': y_test.flatten(),
        'Predicted_SOC': mean_pred,
        'Uncertainty_STD': std_pred
    }).sort_values(by='Time_s')

    results_df.to_csv(predictions_output_path, index=False)

    rms_error = np.sqrt(mean_squared_error(results_df['True_SOC'], results_df['Predicted_SOC']))
    print(f"RMS Error: {rms_error:.6f}")

    if display_results:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['Time_s'], results_df['True_SOC'], label='True SOC', linewidth=2)
        plt.plot(results_df['Time_s'], results_df['Predicted_SOC'], label='Predicted SOC (mean)', linewidth=2)
        plt.fill_between(results_df['Time_s'],
                        results_df['Predicted_SOC'] - results_df['Uncertainty_STD'],
                        results_df['Predicted_SOC'] + results_df['Uncertainty_STD'],
                        alpha=0.3, color='gray', label='Uncertainty (std)')
        plt.xlabel('Time (s)')
        plt.ylabel('SOC')
        plt.title('SOC Estimation with CNN')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model, scaler, results_df, rms_error

def generate_battery_test_data(output_path, I_profile, plot=False):
    model = ECMWithHysteresis(dt=dt, mass=0.4, cp=1050, tau=1000, T_ambient=initial_T_ambient)

    voltage_list, soc_list, current_list, time_list, temperature_list = [], [], [], [], []

    for t, I in enumerate(I_profile):
        if t > 0 and t % (175+300+325) == 0:
            if model.T_ambient < 50:
                model.T_ambient += 2.0
            else:
                model.T_ambient -= 2.0

        
        V_terminal, T_cell = model.step(I, soc)

        if I > 0 and V_terminal < V_min_cutoff:
            I = 0.0
        elif I < 0 and V_terminal > V_max_cutoff:
            I = 0.0

        V_terminal, T_cell = model.step(I, soc)

        voltage_list.append(V_terminal)
        soc_list.append(soc)
        current_list.append(I)
        temperature_list.append(T_cell)
        time_list.append(t)

        soc -= (I * dt) / (capacity_Ah * 3600)
        soc = np.clip(soc, 0.0, 1.0)

    df = pd.DataFrame({
        "Time_s": time_list,
        "Voltage_V": voltage_list,
        "Current_A": current_list,
        "SOC": soc_list,
        "Temperature_C": temperature_list
    })
    df.to_csv(output_path, index=False)

    if plot:
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        axs[0].plot(time_list, voltage_list, label="Voltage [V]", color='tab:blue')
        axs[0].axhline(y=V_min_cutoff, color='red', linestyle='--', label="Min Voltage (2.9 V)")
        axs[0].axhline(y=V_max_cutoff, color='green', linestyle='--', label="Max Voltage (4.4 V)")
        axs[0].set_ylabel("Voltage [V]")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(time_list, current_list, label="Current [A]", color='tab:orange')
        axs[1].set_ylabel("Current [A]")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(time_list, temperature_list, label="Temperature [°C]", color='tab:red')
        axs[2].set_ylabel("Temp [°C]")
        axs[2].legend()
        axs[2].grid(True)

        axs[3].plot(time_list, soc_list, label="SOC", color='tab:purple')
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel("SOC")
        axs[3].legend()
        axs[3].grid(True)

        plt.suptitle("Battery 2RC Simulation")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

def build_interpolator(param, lookup_table):
        soc_vals = np.sort(lookup_table["SOC"].unique())
        temp_vals = np.sort(lookup_table["Temp"].unique())

        pivot = lookup_table.pivot_table(index="Temp", columns="SOC", values=param, aggfunc="mean")
        pivot = pivot.reindex(index=temp_vals, columns=soc_vals)
        pivot = pivot.interpolate(method="linear", axis=0, limit_direction="both")
        pivot = pivot.interpolate(method="linear", axis=1, limit_direction="both")
        pivot = pivot.ffill().bfill().ffill(axis=1).bfill(axis=1)
        pivot = pivot.fillna(1e-3)
        grid = pivot.values
        assert not np.isnan(grid).any(), f"{param} still contains NaNs after filling!"
        return RegularGridInterpolator((temp_vals, soc_vals), grid, bounds_error=False, fill_value=None)

def predict_soc_ekf(data_path, lookup_path, output_path, display_results=False):
    
    df = pd.read_csv(data_path)
    time = df["Time_s"].values
    voltage_meas = df["Voltage_V"].values
    current = df["Current_A"].values
    temp_true = df["Temperature_C"].values
    soc_true = df["SOC"].values

    lookup = pd.read_csv(lookup_path)
    lookup = lookup.groupby(["Temp", "SOC"], as_index=False).mean()

    voc_interp = build_interpolator("OCV", lookup)
    R0_interp = build_interpolator("R0", lookup)
    R1_interp = build_interpolator("R1", lookup)
    C1_interp = build_interpolator("C1", lookup)
    R2_interp = build_interpolator("R2", lookup)
    C2_interp = build_interpolator("C2", lookup)

    voc_from_soc_temp = lambda soc, temp: voc_interp(np.array([[temp, soc]]))[0]
    R0 = lambda soc, temp: R0_interp(np.array([[temp, soc]]))[0]
    R1 = lambda soc, temp: R1_interp(np.array([[temp, soc]]))[0]
    C1 = lambda soc, temp: C1_interp(np.array([[temp, soc]]))[0]
    R2 = lambda soc, temp: R2_interp(np.array([[temp, soc]]))[0]
    C2 = lambda soc, temp: C2_interp(np.array([[temp, soc]]))[0]

    N = len(current)

    # Initial SOC estimation from first voltage
    temp0 = temp_true[0]
    voc0 = voltage_meas[0]

    soc0_guess = np.clip(fsolve(lambda soc: voc_from_soc_temp(float(soc), float(temp0)) - voc0, 0.5)[0], 0.0, 1.0)

    q0 = Qcell * soc0_guess

    print(f"Initial SOC from OCV: {soc0_guess:.3f}, q0 = {q0:.2f} As, T = {temp0:.2f}°C")

    # EKF Initialization 
    z_hat = np.zeros((N, 5))  # [q, vR1, vR2, vh, vR0]
    z_hat[0] = [q0, 0, 0, 0, 0]
    P_tilda = np.zeros((N, 5, 5))
    P_hat = np.zeros((N, 5, 5))
    K = np.zeros((N, 5, 1))
    y_hat = np.zeros(N)

    # Initial P_tilda[0] from bounds
    imax = 100
    r1_0 = R1(soc0_guess, temp0)
    r2_0 = R2(soc0_guess, temp0)
    tau1 = max(r1_0 * C1(soc0_guess, temp0), 1e-3)
    tau2 = max(r2_0 * C2(soc0_guess, temp0), 1e-3)
    dt_rest = 1000
    ez1 = np.exp(-dt_rest / tau1)
    ez2 = np.exp(-dt_rest / tau2)

    zl = np.array([0, -r1_0 * imax * ez1, -r2_0 * imax * ez2, -M, 0])
    zu = np.array([Qcell, r1_0 * imax * ez1, r2_0 * imax * ez2, M, 0])
    P_tilda[0] = np.diag(0.25 * (zu - zl) ** 2)

    # Noise Parameters
    R = np.array([[0.002**2]]) #2mV precision - MAX17843 Chip
    sigma_i = 0.25 # 250mA current measurement precision - Shunt Resistor Sensor
    Qp = np.diag([
        1e-3,   # q
        1e-4,   # vR1
        1e-2,   # vR2
        5e-7,   # vh
        1e-1,   # vR0 
        1e-2,   # gamma
        5e-1,   # M
        1e-1    # eta_tilda
    ])

    # EKF Loop 
    for k in range(1, N):
        i_k = current[k - 1]
        temp_k = temp_true[k - 1]
        eta = eta_tilda if i_k < 0 else 1

        soc_pred = z_hat[k - 1, 0] / Qcell
        r0 = R0(soc_pred, temp_k)
        r1 = R1(soc_pred, temp_k)
        c1 = C1(soc_pred, temp_k)
        r2 = R2(soc_pred, temp_k)
        c2 = C2(soc_pred, temp_k)
        tau1 = max(r1 * c1, 1e-3)
        tau2 = max(r2 * c2, 1e-3)

        e1 = np.exp(-dt / tau1) if dt / tau1 < 700 else 0.0
        e2 = np.exp(-dt / tau2) if dt / tau2 < 700 else 0.0
        eh = np.exp(-abs(eta * i_k * gamma / Qcell) * dt)

        q, vR1_prev, vR2_prev, vh_prev, _ = z_hat[k - 1]
        q_pred = q - eta * i_k * dt
        vR1_pred = e1 * vR1_prev + r1 * (1 - e1) * i_k
        vR2_pred = e2 * vR2_prev + r2 * (1 - e2) * i_k
        vh_pred = eh * vh_prev - M * (1 - eh) * np.sign(i_k)
        vR0_pred = r0 * current[k]
        z_hat[k] = [q_pred, vR1_pred, vR2_pred, vh_pred, vR0_pred]

        dvr1_dR1 = (1 - e1) * i_k
        dvr1_dtau1 = dt * e1 / tau1**2 * (vR1_prev - r1 * i_k)
        dvr2_dR2 = (1 - e2) * i_k
        dvr2_dtau2 = dt * e2 / tau2**2 * (vR2_prev - r2 * i_k)
        dvh_dgamma = -abs(i_k) * dt / Qcell * eh * (vh_prev + M * np.sign(i_k))
        dvh_dM = -(1 - eh) * np.sign(i_k)
        dvh_deta = -abs(i_k * gamma / Qcell) * dt * eh * (vh_prev + M * np.sign(i_k)) if i_k < 0 else 0
        dq_deta = -dt * i_k if i_k < 0 else 0

        J = np.zeros((5, 8))
        J[1, 1] = dvr1_dR1
        J[1, 2] = dvr1_dtau1
        J[2, 3] = dvr2_dR2
        J[2, 4] = dvr2_dtau2
        J[3, 5] = dvh_dgamma
        J[3, 6] = dvh_dM
        J[3, 7] = dvh_deta
        J[0, 7] = dq_deta
        J[4, 0] = i_k

        S = np.diag([sigma_i**2 * i_k**2, 0, sigma_i**2 * current[k]**2])
        B = np.zeros((5, 3))
        B[0, 0] = -eta * dt
        B[1, 0] = r1 * (1 - e1)
        B[2, 0] = r2 * (1 - e2)
        B[3, 0] = -gamma * dt / Qcell * eh * (vh_prev + M * np.sign(i_k)) if i_k < 0 else 0
        B[3, 1] = -M * (1 - eh)
        B[4, 2] = r0
        Q_k = J @ Qp @ J.T + B @ S @ B.T

        dq = 1e-3
        dvoc = (voc_from_soc_temp((q_pred + dq) / Qcell, temp_k) - voc_from_soc_temp((q_pred - dq) / Qcell, temp_k)) / (2 * dq)
        A = np.diag([1, e1, e2, eh, 0])
        C = np.array([[dvoc, -1, -1, 1, -1]])

        P_hat[k] = A @ P_tilda[k - 1] @ A.T + Q_k
        y_hat[k] = voc_from_soc_temp(q_pred / Qcell, temp_k) - vR0_pred - vR1_pred - vR2_pred + vh_pred
        S_k = C @ P_hat[k] @ C.T + R
        K[k] = P_hat[k] @ C.T @ np.linalg.inv(S_k)
        K[k] = np.clip(K[k], -30, 30)
        innovation = voltage_meas[k] - y_hat[k]
        z_hat[k] += (K[k] @ np.array([[innovation]])).flatten()
        P_tilda[k] = (np.eye(5) - K[k] @ C) @ P_hat[k]

    # Postprocessing 
    soc_estimated = np.clip(z_hat[:, 0] / Qcell, 0.0, 1.0)
    df["SOC_EKF"] = soc_estimated

    #  RMSE calculation with NaN-safe masking
    mask = ~np.isnan(soc_true) & ~np.isnan(soc_estimated)

    rmse = np.sqrt(np.mean((soc_estimated[mask] - soc_true[mask])**2))
    print(f"RMSE between EKF and True SOC: {rmse:.6f}")

    df.to_csv(output_path, index=False)

    if display_results:

        # Plotting Results
        plt.figure(figsize=(12, 6))
        plt.plot(time, soc_true, label="True SOC", color="tab:purple")
        plt.plot(time, soc_estimated, label="EKF SOC Estimate", linestyle='--', color="tab:orange")
        plt.fill_between(time,
            np.clip(soc_estimated - 2 * np.sqrt(P_tilda[:, 0, 0]) / Qcell, 0, 1),
            np.clip(soc_estimated + 2 * np.sqrt(P_tilda[:, 0, 0]) / Qcell, 0, 1),
            color='orange', alpha=0.2, label="EKF ±2σ Confidence")
        plt.xlabel("Time [s]")
        plt.ylabel("SOC")
        plt.title(f"True vs EKF Estimated SOC — RMSE = {rmse:.5f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        #  Zoomed-in view (first 10,000 seconds)
        plt.figure(figsize=(10, 5))
        mask = time <= 10000

        plt.plot(time[mask], soc_true[mask], label="True SOC", color="tab:purple")
        plt.plot(time[mask], soc_estimated[mask], label="EKF SOC Estimate", linestyle='--', color="tab:orange")
        plt.fill_between(time[mask],
            np.clip(soc_estimated[mask] - 2 * np.sqrt(P_tilda[mask, 0, 0]) / Qcell, 0, 1),
            np.clip(soc_estimated[mask] + 2 * np.sqrt(P_tilda[mask, 0, 0]) / Qcell, 0, 1),
            color='orange', alpha=0.2, label="EKF ±2σ Confidence")

        plt.xlabel("Time [s]")
        plt.ylabel("SOC")
        plt.title("Zoomed-In: True vs EKF SOC Estimate (First 10,000 s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return df, rmse

def display_comparison_graph(ekf_dataset, cnn_dataset):
    merged_df = pd.merge(cnn_dataset[['Time_s', 'Predicted_SOC', 'Uncertainty_STD', 'True_SOC']], ekf_dataset[['Time_s', 'SOC_EKF']], on='Time_s', how='inner')

    plt.figure(figsize=(12, 6))
    plt.plot(merged_df['Time_s'], merged_df['True_SOC'], label='True SOC', linewidth=2, color='tab:green')
    plt.plot(merged_df['Time_s'], merged_df['Predicted_SOC'], label='CNN SOC Prediction', linewidth=2, color='tab:blue')
    plt.plot(merged_df['Time_s'], merged_df['SOC_EKF'], label='EKF SOC Prediction', linewidth=2, linestyle='--', color='tab:orange')

    plt.fill_between(merged_df['Time_s'],
                     merged_df['Predicted_SOC'] - 2 * merged_df['Uncertainty_STD'],
                     merged_df['Predicted_SOC'] + 2 * merged_df['Uncertainty_STD'],
                     alpha=0.2, color='blue', label='CNN ±2σ Uncertainty')

    plt.xlabel("Time [s]")
    plt.ylabel("State of Charge (SOC)")
    plt.title("Comparison of SOC Estimation: CNN vs EKF vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if generate_simulation_data:
        #Generate Current Profile for simulation data

        I_profile = []
        for _ in range(20):
            I_profile += [40.0] * 175    
            I_profile += [0.0] * 150      
            I_profile += [-18.0] * 325    
            I_profile += [0.0] * 150      

        I_profile += [-25.0] * 800
        I_profile += [0] * 800

        for _ in range(20):
            I_profile += [41.0] * 175     
            I_profile += [0.0] * 150      
            I_profile += [-18.0] * 325    
            I_profile += [0.0] * 150      

        I_profile += [-35.0] * 500
        I_profile += [0] * 800

        for _ in range(20):
            I_profile += [35.0] * 175     
            I_profile += [0.0] * 150      
            I_profile += [-18.0] * 325     
            I_profile += [0.0] * 150      

        I_profile += [-25.0] * 500
        I_profile += [0] * 800

        for _ in range(20):
            I_profile += [37.5] * 175   
            I_profile += [0.0] * 150      
            I_profile += [-18.0] * 325     
            I_profile += [0.0] * 150  

        generate_battery_test_data(simulation_output_path, I_profile=I_profile, plot=True)
    
    if generate_test_data:
        #Generate Current Profile for test data
        I_profile = []

        for _ in range(20):
            I_profile += [35.0] * 175     
            I_profile += [0.0] * 150      
            I_profile += [-18.0] * 325     
            I_profile += [0.0] * 150      

        I_profile += [-25.0] * 500
        I_profile += [0] * 800

        for _ in range(20):
            I_profile += [37.5] * 175   
            I_profile += [0.0] * 150      
            I_profile += [-18.0] * 325     
            I_profile += [0.0] * 150 
        
        generate_battery_test_data(test_data_path, I_profile=I_profile, plot=True)
   
    # Predict SOC using CNN and produce RMS Error
    if to_train_cnn_model:
        model, scaler, results, rms_error_cnn = train_cnn_model(simulation_output_path, model_output_path, scaler_output_path, display_results=True, cnn_predictions_output_path=cnn_predictions_output_path)
    else:
        model, scaler = load_cnn_model_from_file(model_output_path, scaler_output_path)
        results_cnn, rms_error_cnn = predict_with_cnn(simulation_output_path, model, scaler, cnn_predictions_output_path, display_results=True)

    # Predict SOC using EKF Filter - Works
    results_ekf, rms_error_ekf = predict_soc_ekf(simulation_output_path, lookup_table_path, ekf_predictions_output_path, display_results=True)

    # Display graph that compares both techniques with respect to the Real SOC.
    display_comparison_graph(results_ekf, results_cnn)
