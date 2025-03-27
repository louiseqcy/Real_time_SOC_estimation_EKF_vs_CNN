import pybamm
 
pybamm.set_logging_level("INFO")

# Set up a Thevenin model with 3 RC elements
options = {"number of rc elements": 2}
model = pybamm.equivalent_circuit.Thevenin(options=options)

# Load default parameter values
parameter_values = model.default_parameter_values

# Update resistance and capacitance values for the 3 RC elements
parameter_values.update(
    {
        "R0 [Ohm]": 0.0012, 
        "R1 [Ohm]": 0.005,
        "R2 [Ohm]": 0.001,
        "C1 [F]": 1000,
        "C2 [F]": 5000,
        "Nominal cell capacity [A.h]": 14,  # Required for "C/10" 14000mAh
        "Initial State of Charge": 2.0,
        "Lower voltage cut-off [V]": 3,
        "Upper voltage cut-off [V]": 4.35,
        "Element-2 initial overpotential [V]": 0.0,
    },
    check_already_exists=False,
)
 
# Define the experiment
experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/2 for 10 hours or until 3.0 V",
            "Rest for 1 hour",
            "Charge at 1 A until 4.1 V",
            "Hold at 4.1 V until 50 mA",
            "Rest for 1 hour",
        ),
    ]
)
 
# Run the simulation
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
sim.solve()
sim.plot()