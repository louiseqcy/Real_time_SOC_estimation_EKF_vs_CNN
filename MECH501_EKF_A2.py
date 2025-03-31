import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# === Load data ===
df = pd.read_csv("battery_sim_output.csv")
time = df["Time_s"].values
voltage_meas = df["Voltage_V"].values
current = df["Current_A"].values
temp_true = df["Temperature_C"].values

# === Load OCV-SOC-Temp lookup ===
lookup = pd.read_csv("battery_lookup_table.csv")
lookup = lookup.groupby(["Temp", "SOC"], as_index=False).mean()  # clean duplicates

soc_vals = np.sort(lookup["SOC"].unique())
temp_vals = np.sort(lookup["Temp"].unique())

def build_interpolator(param):
    pivot = lookup.pivot_table(index="Temp", columns="SOC", values=param, aggfunc="mean")
    pivot = pivot.reindex(index=temp_vals, columns=soc_vals)
    pivot = pivot.interpolate(method="linear", axis=0, limit_direction="both")
    pivot = pivot.interpolate(method="linear", axis=1, limit_direction="both")
    pivot = pivot.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1)
    grid = pivot.values
    return RegularGridInterpolator((temp_vals, soc_vals), grid, bounds_error=False, fill_value=None)

# Interpolated model parameters
voc_from_soc_temp = lambda soc, temp: voc_interp(np.array([[temp, soc]]))[0]
R0 = lambda soc, temp: R0_interp(np.array([[temp, soc]]))[0]
R1 = lambda soc, temp: R1_interp(np.array([[temp, soc]]))[0]
C1 = lambda soc, temp: C1_interp(np.array([[temp, soc]]))[0]
R2 = lambda soc, temp: R2_interp(np.array([[temp, soc]]))[0]
C2 = lambda soc, temp: C2_interp(np.array([[temp, soc]]))[0]

voc_interp = build_interpolator("OCV")
R0_interp = build_interpolator("R0")
R1_interp = build_interpolator("R1")
C1_interp = build_interpolator("C1")
R2_interp = build_interpolator("R2")
C2_interp = build_interpolator("C2")

# === Constants ===
Qcell = 14000  # Coulombs
gamma, M = 1, 1
eta_tilda = 0.98
dt = 1.0
N = len(current)

# === Initial SOC estimation from first voltage (assume rest) ===
temp0 = temp_true[0]
voc0 = voltage_meas[0]

def voc_diff(soc):  # wrapper to fix fsolve
    return voc_from_soc_temp(float(soc), float(temp0)) - voc0

soc0_guess = fsolve(voc_diff, 0.5)[0]
soc0_guess = np.clip(soc0_guess, 0.0, 1.0)
q0 = Qcell * soc0_guess
print(f"Initial SOC from v_OC: {soc0_guess:.3f}, q0 = {q0:.2f} As, T = {temp0:.2f}°C")

# === EKF state initialization ===
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
tau1 = r1_0 * C1(soc0_guess, temp0)
tau2 = r2_0 * C2(soc0_guess, temp0)
dt_rest = 1000
ez1 = np.exp(-dt_rest / tau1)
ez2 = np.exp(-dt_rest / tau2)

zl = np.array([0, -r1_0 * imax * ez1, -r2_0 * imax * ez2, -M, 0])
zu = np.array([Qcell, r1_0 * imax * ez1, r2_0 * imax * ez2, M, 0])
P_tilda[0] = np.diag(0.25 * (zu - zl) ** 2)

# === Noise parameters ===
R = np.array([[0.01**2]])  # Measurement noise
sigma_i = 0.02  # Input noise
Qp = np.diag([
    1e-7, 1e-7, 1e-3, 1e-7, 1e-3, 1e-2, 1e-2, 1e-2  # Param variances
])

# === EKF Loop ===
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
    q_pred = q - eta * i_k * dt
    vR1_pred = e1 * vR1_prev + r1 * (1 - e1) * i_k
    vR2_pred = e2 * vR2_prev + r2 * (1 - e2) * i_k
    vh_pred = eh * vh_prev - M * (1 - eh) * np.sign(i_k)
    vR0_pred = r0 * current[k]
    z_hat[k] = [q_pred, vR1_pred, vR2_pred, vh_pred, vR0_pred]

    # === Jacobians ===
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

    # === Linearization for correction ===
    dq = 1e-3
    dvoc = (voc_from_soc_temp((q_pred + dq) / Qcell, temp_k) -
            voc_from_soc_temp((q_pred - dq) / Qcell, temp_k)) / (2 * dq)
    A = np.diag([1, e1, e2, eh, 0])
    C = np.array([[dvoc, -1, -1, 1, -1]])

    # === EKF predict & correct ===
    P_hat[k] = A @ P_tilda[k - 1] @ A.T + Q_k
    y_hat[k] = voc_from_soc_temp(q_pred / Qcell, temp_k) - vR0_pred - vR1_pred - vR2_pred + vh_pred
    S_k = C @ P_hat[k] @ C.T + R
    K[k] = P_hat[k] @ C.T @ np.linalg.inv(S_k)
    innovation = voltage_meas[k] - y_hat[k]
    z_hat[k] += (K[k] @ np.array([[innovation]])).flatten()
    P_tilda[k] = (np.eye(5) - K[k] @ C) @ P_hat[k]

# === Output & Plot ===
soc_estimated = np.clip(z_hat[:, 0] / Qcell, 0.0, 1.0)
df["SOC_EKF"] = soc_estimated

plt.figure(figsize=(12, 6))
plt.plot(df["Time_s"], df["SOC_EKF"], label="EKF SOC Estimate", linestyle='--', color="tab:orange")
plt.fill_between(df["Time_s"],
                 np.clip(df["SOC_EKF"] - 2 * np.sqrt(P_tilda[:, 0, 0]) / Qcell, 0, 1),
                 np.clip(df["SOC_EKF"] + 2 * np.sqrt(P_tilda[:, 0, 0]) / Qcell, 0, 1),
                 color='orange', alpha=0.2, label="EKF ±2σ Confidence")
plt.xlabel("Time [s]")
plt.ylabel("SOC")
plt.title("EKF Estimated SOC with Confidence Interval")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()