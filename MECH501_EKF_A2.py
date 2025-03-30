import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, RegularGridInterpolator
import matplotlib.pyplot as plt

# Load simulation output
df = pd.read_csv("battery_sim_output.csv")
time = df["Time_s"].values
voltage_meas = df["Voltage_V"].values
current = df["Current_A"].values
soc_true = df["SOC"].values
temp_true = df["Temperature_C"].values

# Load and clean lookup table
lookup = pd.read_csv("battery_lookup_table.csv")
lookup = lookup.groupby(["Temp", "SOC"], as_index=False).mean()  # Remove duplicates

# Unique grid points
soc_vals = np.sort(lookup["SOC"].unique())
temp_vals = np.sort(lookup["Temp"].unique())

# Build interpolators
def build_interpolator(param):
    pivot = lookup.pivot_table(index="Temp", columns="SOC", values=param, aggfunc="mean")
    pivot = pivot.reindex(index=temp_vals, columns=soc_vals)
    pivot = pivot.interpolate(method="linear", axis=0, limit_direction="both")
    pivot = pivot.interpolate(method="linear", axis=1, limit_direction="both")
    pivot = pivot.ffill(axis=0).bfill(axis=0)
    pivot = pivot.ffill(axis=1).bfill(axis=1)
    grid = pivot.values
    assert grid.shape == (len(temp_vals), len(soc_vals)), f"Grid shape mismatch: {grid.shape}"
    return RegularGridInterpolator((temp_vals, soc_vals), grid, bounds_error=False, fill_value=None)

# Create interpolated functions
def param_fn(interp):
    return lambda soc, temp: interp(np.array([[float(temp), float(soc)]]))[0]

voc_from_soc_temp = param_fn(build_interpolator("OCV"))
R0 = param_fn(build_interpolator("R0"))
R1 = param_fn(build_interpolator("R1"))
C1 = param_fn(build_interpolator("C1"))
R2 = param_fn(build_interpolator("R2"))
C2 = param_fn(build_interpolator("C2"))

# Constants
Qcell = 14000  # Coulombs
gamma, M = 1, 1
eta_tilda = 0.98
dt = 1.0
N = len(current)

# EKF Initialization
z_hat = np.zeros((N, 5))  # [q, vR1, vR2, vh, vR0]
P_tilda = np.zeros((N, 5, 5))
P_hat = np.zeros((N, 5, 5))
K = np.zeros((N, 5, 1))
y_hat = np.zeros(N)

# Estimate initial SOC from first voltage
voc0 = voltage_meas[0]
temp0 = temp_true[0]
soc0_guess = fsolve(lambda soc: voc_from_soc_temp(soc, temp0) - voc0, 0.8)[0]
soc0_guess = np.clip(soc0_guess, 0.0, 1.0)
q0 = Qcell * soc0_guess
z_hat[0] = [q0, 0, 0, 0, 0]
P_tilda[0] = np.diag([1e3, 1e-4, 1e-4, 1e-2, 1e-4])

# Measurement noise
R = np.array([[0.01**2]])

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
    tau1, tau2 = r1 * c1, r2 * c2

    e1 = np.exp(-dt / tau1)
    e2 = np.exp(-dt / tau2)
    eh = np.exp(-abs(eta * i_k * gamma / Qcell) * dt)
    q, vR1_prev, vR2_prev, vh_prev, _ = z_hat[k - 1]

    # State prediction
    q_pred = q - eta * i_k * dt
    vR1_pred = e1 * vR1_prev + r1 * (1 - e1) * i_k
    vR2_pred = e2 * vR2_prev + r2 * (1 - e2) * i_k
    vh_pred = eh * vh_prev - M * (1 - eh) * np.sign(i_k)
    vR0_pred = r0 * current[k]
    z_hat[k] = [q_pred, vR1_pred, vR2_pred, vh_pred, vR0_pred]

    # Linearization
    dq = 1e-3
    dvoc = (voc_from_soc_temp((q_pred + dq) / Qcell, temp_k) - voc_from_soc_temp((q_pred - dq) / Qcell, temp_k)) / (2 * dq)
    A = np.diag([1, e1, e2, eh, 0])
    C = np.array([[dvoc, -1, -1, 1, -1]])
    Q_k = np.diag([0.01, 1e-4, 1e-4, 1e-4, 1e-5])

    # Predict and correct
    P_hat[k] = A @ P_tilda[k - 1] @ A.T + Q_k
    y_hat[k] = voc_from_soc_temp(q_pred / Qcell, temp_k) - vR0_pred - vR1_pred - vR2_pred + vh_pred
    S_k = C @ P_hat[k] @ C.T + R
    K[k] = P_hat[k] @ C.T @ np.linalg.inv(S_k)
    innovation = voltage_meas[k] - y_hat[k]
    z_hat[k] += (K[k] @ np.array([[innovation]])).flatten()
    P_tilda[k] = (np.eye(5) - K[k] @ C) @ P_hat[k]

# Output and plot
soc_estimated = np.clip(z_hat[:, 0] / Qcell, 0.0, 1.0)
df["SOC_EKF"] = soc_estimated

plt.figure(figsize=(12, 6))
plt.plot(df["Time_s"], df["SOC"], label="True SOC", color="tab:purple")
plt.plot(df["Time_s"], df["SOC_EKF"], label="EKF SOC Estimate", linestyle='--', color="tab:orange")
plt.xlabel("Time [s]")
plt.ylabel("SOC")
plt.title("True vs EKF Estimated SOC (Temperature + SOC Dependent)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()