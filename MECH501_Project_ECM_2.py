import pybamm

options = {"number of rc elements": 2}
model = pybamm.equivalent_circuit.Thevenin(options=options)

parameter_values = model.default_parameter_values
parameter_values.update({
    'Initial SoC': 1.0, 
    'Lower voltage cut-off [V]': 3.05,
    'Upper voltage cut-off [V]': 4.3,
    'Cell capacity [A.h]': 14.5,
    'Nominal cell capacity [A.h]': 14.0,
    'R0 [Ohm]': 0.0035,
    'R1 [Ohm]': 0.0076,
    'C1 [F]': 6761,
    'R2 [Ohm]': 1e-6,
    'C2 [F]': 1260,
    'Element-2 initial overpotential [V]': 0,
    'Open-circuit voltage at 0% SOC [V]': 3.0,
    'Open-circuit voltage at 100% SOC [V]': 4.35
}, check_already_exists=False)

experiment = pybamm.Experiment(
    ["Discharge at 2C for 15 minutes or until 3.2 V", 
     'Rest for 0.5 hours', 
     'Charge at 1C until 4.3V'] * 5, period='1 second'
)

sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
sim.solve([0, 3600*20])
sim.plot()