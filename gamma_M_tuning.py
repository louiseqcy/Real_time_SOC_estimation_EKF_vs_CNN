import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === Load data ===
df = pd.read_csv("battery_sim_output.csv")
time = df["Time_s"].values
voltage_meas = df["Voltage_V"].values
current = df["Current_A"].values
temp_true = df["Temperature_C"].values
soc_true = df["SOC"].values

# === Load and clean lookup table ===
lookup = pd.read_csv("battery_lookup_table.csv")
lookup = lookup.groupby(["Temp", "SOC"], as_index=False).mean()
soc_vals = np.sort(lookup["SOC"].unique())
temp_vals = np.sort(lookup["Temp"].unique())

def build_interpolator(param):
    pivot = lookup.pivot_table(index="Temp", columns="SOC", values=param, aggfunc="mean")
    pivot = pivot.reindex(index=temp_vals, columns=soc_vals)
    pivot = pivot.interpolate(method="linear", axis=0, limit_direction="both")
    pivot = pivot.interpolate(method="linear", axis=1, limit_direction="both")
    pivot = pivot.ffill().bfill().ffill(axis=1).bfill(axis=1)
    pivot = pivot.fillna(method='ffill').fillna(method='bfill')
    if pivot.isna().any().any():
        pivot = pivot.fillna(1e-3)
    grid = pivot.values
    return RegularGridInterpolator((temp_vals, soc_vals), grid, bounds_error=False, fill_value=None)

voc_interp = build_interpolator("OCV")
R0_interp = build_interpolator("R0")
R1_interp = build_interpolator("R1")
C1_interp = build_interpolator("C1")
R2_interp = build_interpolator("R2")
C2_interp = build_interpolator("C2")

voc_from_soc_temp = lambda soc, temp: voc_interp(np.array([[temp, soc]]))[0]
R0 = lambda soc, temp: R0_interp(np.array([[temp, soc]]))[0]
R1 = lambda soc, temp: R1_interp(np.array([[temp, soc]]))[0]
C1 = lambda soc, temp: C1_interp(np.array([[temp, soc]]))[0]
R2 = lambda soc, temp: R2_interp(np.array([[temp, soc]]))[0]
C2 = lambda soc, temp: C2_interp(np.array([[temp, soc]]))[0]

# === Constants ===
Qcell = 14 * 3600  # 14 Ah to As
dt = 1.0
N = len(current)
eta_tilda = 0.98

def run_ekf(M, gamma):
    temp0 = temp_true[0]
    voc0 = voltage_meas[0]
    soc0_guess = fsolve(lambda soc: voc_from_soc_temp(float(soc), float(temp0)) - voc0, 0.5)[0]
    soc0_guess = np.clip(soc0_guess, 0.0, 1.0)
    q0 = Qcell * soc0_guess

    z_hat = np.zeros((N, 5))
    z_hat[0] = [q0, 0, 0, 0, 0]
    P_tilda = np.zeros((N, 5, 5))
    P_hat = np.zeros((N, 5, 5))
    K = np.zeros((N, 5, 1))
    y_hat = np.zeros(N)

    r1_0 = R1(soc0_guess, temp0)
    r2_0 = R2(soc0_guess, temp0)
    tau1 = r1_0 * C1(soc0_guess, temp0)
    tau2 = r2_0 * C2(soc0_guess, temp0)
    dt_rest = 1000
    ez1 = np.exp(-dt_rest / tau1)
    ez2 = np.exp(-dt_rest / tau2)
    imax = 100
    zl = np.array([0, -r1_0 * imax * ez1, -r2_0 * imax * ez2, -M, 0])
    zu = np.array([Qcell, r1_0 * imax * ez1, r2_0 * imax * ez2, M, 0])
    P_tilda[0] = np.diag(0.25 * (zu - zl) ** 2)

    R = np.array([[0.01 ** 2]])
    sigma_i = 0.02
    Qp = np.diag([1e-7, 1e-7, 1e-3, 1e-7, 1e-3, 1e-2, 1e-2, 1e-2])

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

        dq = 1e-3
        dvoc = (voc_from_soc_temp((q_pred + dq) / Qcell, temp_k) -
                voc_from_soc_temp((q_pred - dq) / Qcell, temp_k)) / (2 * dq)
        A = np.diag([1, e1, e2, eh, 0])
        C = np.array([[dvoc, -1, -1, 1, -1]])

        P_hat[k] = A @ P_tilda[k - 1] @ A.T + np.eye(5) * 1e-6  # Q_k small fudge
        y_hat[k] = voc_from_soc_temp(q_pred / Qcell, temp_k) - vR0_pred - vR1_pred - vR2_pred + vh_pred
        S_k = C @ P_hat[k] @ C.T + R
        K[k] = P_hat[k] @ C.T @ np.linalg.inv(S_k)
        innovation = voltage_meas[k] - y_hat[k]
        z_hat[k] += (K[k] @ np.array([[innovation]])).flatten()
        P_tilda[k] = (np.eye(5) - K[k] @ C) @ P_hat[k]

    soc_estimated = np.clip(z_hat[:, 0] / Qcell, 0.0, 1.0)
    rmse = np.sqrt(mean_squared_error(soc_true, soc_estimated))
    return rmse

# === Grid search ===
M_vals = np.linspace(0.1, 1.5, 5)
gamma_vals = np.linspace(0.1, 2.0, 5)

best_rmse = np.inf
best_M, best_gamma = None, None

print("Tuning M and gamma:")
for M in M_vals:
    for gamma in gamma_vals:
        try:
            rmse = run_ekf(M, gamma)
            print(f"M = {M:.2f}, gamma = {gamma:.2f} => RMSE = {rmse:.5f}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_M, best_gamma = M, gamma
        except Exception as e:
            print(f"M = {M:.2f}, gamma = {gamma:.2f} => ERROR: {e}")

print("\n=== Best Parameters ===")
print(f"Best M = {best_M:.3f}, Best gamma = {best_gamma:.3f} → RMSE = {best_rmse:.5f}")