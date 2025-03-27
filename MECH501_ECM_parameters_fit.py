import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# RC Voltage drop function for an RC branch
def f_rc(V, I, R, C):
    return -V / (R * C) + I / C

# RK3 integration method
def rk3_step(f, xk, tk, h, *args):
    k1 = f(xk, tk, *args)
    k2 = f(xk + (h / 2) * k1, tk + h / 2, *args)
    k3 = f(xk + h * k2, tk + h, *args)
    return xk + (h / 6) * (k1 + 4 * k2 + k3)

# ECM simulation that now accepts a temperature argument
def simulate_ecm(params, current, T, dt=1, Q_nom=14000, T_ref=25):
    # Initialize & Unpack parameters
    R0, R1, C1, R2, C2, ocv_temp_coeff, Q = params
    N = len(current)

    V1 = np.zeros(N)
    V2 = np.zeros(N)
    SOC = np.zeros(N)
    V_terminal = np.zeros(N)
    
    # Initial conditions
    V1_k, V2_k, soc_k = 0.0, 0.0, 1.0
    
    for k in range(N):
        I_k = current[k]
        t_k = k * dt
        
        # Update state of charge (SOC) using the current profile
        soc_k -= (I_k * dt) / Q_nom
        SOC[k] = soc_k
        
        # OCV with a temperature correction:
        # and ocv_temp_coeff adjusts for deviation from the reference temperature (T_ref)
        ocv_k = (3.0 + 1.2 * soc_k) + ocv_temp_coeff * (T - T_ref)
        
        # Update RC branch voltages using RK3 integration
        V1_k = rk3_step(f_rc, V1_k, t_k, dt, I_k, R1, C1)
        V2_k = rk3_step(f_rc, V2_k, t_k, dt, I_k, R2, C2)
        
        # Terminal voltage of the ECM
        V_terminal[k] = ocv_k - R0 * I_k - V1_k - V2_k
        
    return V_terminal, SOC

# Compute measured SOC from capacity data.
# Assumes "Capacity (mAh)" is the cumulative discharged capacity,
# and Q_nominal is the nominal capacity.
def compute_measured_SOC(capacity, Q_nominal):
    return 1 - capacity / Q_nominal

# Combined objective function that minimizes errors in voltage and SOC
def objective_function_combined(params, datasets, dt=1, T_ref=25, weight_voltage=1.0, weight_soc=0.1):
    total_error = 0
    for data in datasets:
        current = data["current"]
        measured_voltage = data["voltage"]
        measured_capacity = data["capacity"]
        T = data["temperature"]
        
        simulated_voltage, simulated_SOC = simulate_ecm(params, current, T, dt, T_ref)
        # Use Q from the parameters as the nominal capacity for SOC computation
        Q_nominal = params[-1]
        measured_SOC = compute_measured_SOC(measured_capacity, Q_nominal)
        
        error_voltage = np.sum((simulated_voltage - measured_voltage)**2)
        error_soc = np.sum((simulated_SOC - measured_SOC)**2)
        total_error += weight_voltage * error_voltage + weight_soc * error_soc
    return total_error

# Load your CSV files.
data1C = pd.read_csv("Melasta_1C_Discharge.csv")
data5C = pd.read_csv("Melasta_5C_Discharge.csv")
data10C = pd.read_csv("Melasta_10C_Discharge.csv")

# Organize datasets with proper column names.
datasets = []
datasets.append({
    "current": data1C["Current (A)"].values, 
    "voltage": data1C["Voltage (V)"].values, 
    "capacity": data1C["Capacity (mAh)"].values, 
    "temperature": data1C["T1(C)"].values[0]
})
datasets.append({
    "current": data5C["Current (A)"].values, 
    "voltage": data5C["Voltage (V)"].values, 
    "capacity": data5C["Capacity (mAh)"].values, 
    "temperature": data5C["T1(C)"].values[0]
})
datasets.append({
    "current": data10C["Current (A)"].values, 
    "voltage": data10C["Voltage (V)"].values, 
    "capacity": data10C["Capacity (mAh)"].values, 
    "temperature": data10C["T1(C)"].values[0]
})

# Initial guess for parameters: [R0, R1, C1, R2, C2, ocv_temp_coeff, Q]
initial_params = [0.01, 0.01, 1000, 0.01, 1000, 0.0, 1750]

# Optimize the parameters using Nelder-Mead algorithm.
result = minimize(objective_function_combined, initial_params, args=(datasets,), method='Nelder-Mead')
print("Fitted parameters:", result.x)
print("Objective function value:", result.fun)

# Plot the simulated vs. measured voltage for each dataset.
fig, axes = plt.subplots(3, 1, figsize=(10, 12))
dataset_names = ['1C', '5C', '10C']
for i, data in enumerate(datasets):
    current = data["current"]
    measured_voltage = data["voltage"]
    measured_capacity = data["capacity"]
    T = data["temperature"]
    
    simulated_voltage, simulated_SOC = simulate_ecm(result.x, current, T)
    Q_nominal = result.x[-1]
    measured_SOC = compute_measured_SOC(measured_capacity, Q_nominal)
    
    axes[i].plot(simulated_voltage, label='Simulated Voltage')
    axes[i].plot(measured_voltage, label='Measured Voltage', linestyle='--')
    axes[i].set_title(f"ECM Fit for {dataset_names[i]} Discharge at {T}°C")
    axes[i].set_xlabel("Time step")
    axes[i].set_ylabel("Voltage (V)")
    axes[i].legend()

plt.tight_layout()
plt.show()