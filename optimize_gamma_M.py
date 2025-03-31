
import numpy as np
import subprocess
import re

# === Parameters ===
learning_rate = 0.01
max_iters = 100
tolerance = 1e-4

# === Initial guesses ===
gamma = 1.0
M = 0.4

# Logistic function for constraining gamma and M
def logistic(x):
    return 1 / (1 + np.exp(-x))

def inverse_logistic(y):
    return np.log(y / (1 - y))

# === Transform parameters to unconstrained space ===
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
