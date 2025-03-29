import numpy as np
import math
import sympy as sp

# All constants
eta = 1
eta_tilda = 1
Tau1 = 1
Tau2 = 1
dt = 1
R0 = 1
R1 = 1
R2 = 1
gamma = 1
Qcell = 14000
N = 1
M = 1 # not sure

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

# #%% Predict
# Prediction of the system state
z_hat[k+1] = z(k+1)

# Prediction of the system state error
P_hat[k+1] = A[k+1] @ P_tilda[k] @ A[k+1].T + Q[k]

# Prediction of the measured value
y[k+1] = y(k+1, z_hat[k+1])

# Kalman gain
K[k+1] = P_hat[k+1] @ C[k+1].T / (C[k+1] @ P_hat[k+1] @ C[k+1].T + R[k+1] )

#%% Correction Update
# Correction of the system state
z_tilda[k+1] = z_hat[k+1] + K[k+1] @ (v[k+1] - y_hat[k+1])

# Correction of the system state error
P_tilda[k+1] = (np.ones(N,N) - K[k+1] @ C[k+1]) @ P_hat[k] 


