import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve

k0 = 0.005   
k1 = 0.0002    
k2 = 0.01      

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

capacity_Ah = 14.0
dt = 1.0
V_min_cutoff = 2.9
V_max_cutoff = 4.4
soc = 0.9
initial_T_ambient = 25.0
model = ECMWithHysteresis(dt=dt, mass=0.4, cp=1050, tau=1000, T_ambient=initial_T_ambient)

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
df.to_csv("battery_sim_output.csv", index=False)

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

plt.suptitle("Battery Simulation: Hysteresis + Heating Environment")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
