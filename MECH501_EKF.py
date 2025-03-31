import numpy as np
import math
import sympy as sp

# All constants

# Battery parameters (example values)
R0, R1, R2 = 0.0035, 0.007, 0.0001       # Ohms
C1, C2 = 6761, 1206                      # Farads
Tau1, Tau2 = R1*C1, R2*C2
gamma, M = 1, 1
Qcell = 14_000                           # As per your context
eta_tilda = 0.98
dt = 1    

q = np.ones(N) # Charge in the cell
z = np.ones(N) # SOC: z = q / Qcell
vOCV = np.ones(N) # Open-circuit-voltage (OCV)
vR0 = np.ones(N) # Voltage at internal resistance
vR1 = np.ones(N) # Voltage at the 1st RC element
vR2 = np.ones(N) # Voltage at the 2nd RC element
vh = np.ones(N) # hysteresis voltage
dt_rest = 1 # Resting time before initialization
R = np.ones(N,N) # Measurement noise covariance
Q = np.ones(N,N) # Process noise covariance
S = np.ones(N,N) # Input noise covariance

for k in range(N):
    i[k] = current[k]

    if i[k] < 0:
        eta[k] = eta_tilda
    else:
        eta[k] = 1

# State Equations z = ( q[k], vR1[k], vR2[k], vh[k], vRo[k] )
def z(k):
    q = q[k-1] - eta[k-1] * i[k-1] * dt # (2)
    vR1 = e1 * vR1[k-1] + R1 * (1 - e1) * i[k-1] # (3)
    vR1 = e2 * vR2[k-1] + R2 * (1 - e2) * i[k-1] # (3)
    vh = math.exp(-abs(eta[k-1] * i[k-1] * gamma / Qcell) * dt) * vh[k-1] - M * (1 - math.exp(-abs(eta[k-1] * i[k-1] * gamma / Qcell) * dt)) * sp.sign(i[k-1]) # (4)
    vR0 = R0 * i[k] # (5)

    z = np.array([q, vR1, vR2, vh, vR0])
    
    return z

# Measurement Equation
def y(k, z):
    q, vR1, vR2, vh, vR0 = z 
    v = vOCV(q) - vR0 - vR1 - vR2 + vh
    return v

# Matrices
e1 = math.exp(-dt/Tau1)
e2 = math.exp(-dt/Tau2)

def eh(k):
    eh = math.exp(-abs(eta[k] * i[k] * gamma / Qcell) * dt)
    return eh

def A(k):
    A = np.diag(np.array([1, e1, e2, eh(k), 0]))
    return A

def B(k):
    B = np.array([-eta[k-1]*dt, 0 , 0],
                 [R1(1 - e1), 0 , 0],
                 [R2(1 - e1), 0 , 0],
                 [-eta * gamma * dt / Qcell * (np.sign(i[k-1]) * vh[k-1] + M * eh(k)), -M * (1 - eh(k)), 0],
                 [0, 0, R0])
    return B

def C(k, z):
    q, _, _, _, _ = z 
    C = [vOC.diff(q), -1, -1, 1, -1]

    return C

#%% Initialization 
# Initilization of the system state
vR1[0] = 0
vR2[0] = 0
vh[0] = 0
vR0[0] = 0
q0 = # chosen such that vOC[q0] = v[0]

z_tilda[0] = [q0, vR1[0], vR2[0], vh[0], vR0[0]] # after long resting time, this is the best estimation for the initial state

# Initilization of the system state error
P_tilda[0] = np.diag()

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


