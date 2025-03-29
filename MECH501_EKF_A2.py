import numpy as np
import math
from scipy.interpolate import interp1d

# Battery parameters (example values)
R0, R1, R2 = 0.0035, 0.007, 0.0001       # Ohms
C1, C2 = 6761, 1206                      # Farads
Tau1, Tau2 = R1*C1, R2*C2
gamma, M = 1, 1
Qcell = 14_000                           # As per your context
eta_tilda = 0.98
dt = 1                                   # Time step in seconds

# Time step count
N = 100
current = np.zeros(N)                   # Fill with your current data
voltage_meas = np.zeros(N)             # Fill with your measured voltages

# Create OCV function (should come from your OCV-SOC data)
SOC_vals = np.linspace(0, 1, 100)
OCV_vals = 3 + 1.35 * SOC_vals          # Example linear relationship
voc = interp1d(Qcell * SOC_vals, OCV_vals, fill_value="extrapolate")

# Measurement noise
sigma_v = 0.01                          # e.g., voltage sensor noise
R = np.eye(1) * sigma_v**2

# Initializations
z_hat = np.zeros((N, 5))               # States: [q, vR1, vR2, vh, vR0]
P_hat = np.zeros((N, 5, 5))
P_tilda = np.zeros((N, 5, 5))
K = np.zeros((N, 5, 1))
y_hat = np.zeros(N)

# Initial state
q0 = Qcell * 0.8
z_hat[0] = [q0, 0, 0, 0, 0]
P_tilda[0] = np.diag([5, 0.001, 0.001, 0.01, 0.001])  # Initial covariances

# EKF loop
for k in range(1, N):
    i_k = current[k-1]
    eta = eta_tilda if i_k < 0 else 1

    # Precompute decays
    e1 = math.exp(-dt / Tau1)
    e2 = math.exp(-dt / Tau2)
    eh = math.exp(-abs(eta * i_k * gamma / Qcell) * dt)

    # Previous state
    q, vR1, vR2, vh, _ = z_hat[k-1]

    # State Prediction
    q_pred = q - eta * i_k * dt
    vR1_pred = e1 * vR1 + R1 * (1 - e1) * i_k
    vR2_pred = e2 * vR2 + R2 * (1 - e2) * i_k
    vh_pred = eh * vh - M * (1 - eh) * np.sign(i_k)
    vR0_pred = R0 * current[k]

    z_hat[k] = [q_pred, vR1_pred, vR2_pred, vh_pred, vR0_pred]

    # A matrix
    A = np.diag([1, e1, e2, eh, 0])

    # C matrix (Jacobian of measurement function)
    dvoc_dq = (voc(q_pred + 1e-3) - voc(q_pred - 1e-3)) / 2e-3  # Numerical derivative
    C = np.array([[dvoc_dq, -1, -1, 1, -1]])

    # Predict measurement
    y_hat[k] = voc(q_pred) - vR0_pred - vR1_pred - vR2_pred + vh_pred

    # Process noise (tune or compute via J·Qp·Jᵀ + B·S·Bᵀ)
    Q = np.diag([0.01, 1e-4, 1e-4, 1e-4, 1e-5])

    # Predict covariance
    P_hat[k] = A @ P_tilda[k-1] @ A.T + Q

    # Kalman gain
    S_k = C @ P_hat[k] @ C.T + R
    K[k] = P_hat[k] @ C.T @ np.linalg.inv(S_k)

    # Correction step
    z_hat[k] += (K[k] @ (voltage_meas[k] - y_hat[k])).flatten()
    P_tilda[k] = (np.eye(5) - K[k] @ C) @ P_hat[k]

# Convert final q to SOC
SOC_estimated = z_hat[:, 0] / Qcell