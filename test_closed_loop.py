# test_closed_loop.py
# Implements Phase 2, Step 4: The Closed-Loop Simulation
# This script runs Scenario 1: Stabilize from 5 degrees.
# Includes FIX for solver failure and 'AttributeError'.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Import our classes from the other files
from robot_model import RobotModel
from controller import Controller

# --- 1. Setup Simulation ---

print("Setting up simulation...")
robot = RobotModel()
controller = Controller()

# Simulation parameters
T_START = 0.0
T_END = 15.0  # Run for 15s to match Figure 11
DT = 0.01     # Time step for controller update [s]

# --- 2. Setup Scenario 1: Stabilize from 5 degrees ---
# (This is to replicate Figure 11)

# Initial state [x1, x2, x3, x4]
theta_0 = 5.0 * (np.pi / 180.0)  # 5.0 degrees [rad]
theta_dot_0 = 0.0
x_0 = 0.0
x_dot_0 = 0.0
y_current = np.array([theta_0, theta_dot_0, x_0, x_dot_0])

# Reference state [x1_ref, x3_ref]
theta_ref = 0.0 * (np.pi / 180.0) # 0 degrees
x_ref = 0.0                       # 0 meters
y_ref = np.array([theta_ref, x_ref])

# Reset controller states (like integral term z1)
controller.reset_states()

# --- FIX: Numerical Stability ---
# Scale down the position gains to prevent the solver from failing.
# The original gains are too aggressive and "fight" the balance controller.
GAIN_SCALE = 0.1 
controller.Kp_pos *= GAIN_SCALE
controller.Kd_pos *= GAIN_SCALE
print(f"Applying position gain scale: {GAIN_SCALE}")
# --- End of Fix ---


# --- 3. Run Simulation Loop ---

# Create time array
t_eval = np.arange(T_START, T_END, DT)
num_steps = len(t_eval)

# Prepare arrays to store the results
# We'll store 4 state variables + 1 time variable
history = np.zeros((num_steps, 5))

print(f"Running closed-loop simulation for {T_END} seconds...")

for i in range(num_steps):
    t = t_eval[i]
    
    # Store current state and time
    history[i, 0] = t
    history[i, 1:] = y_current
    
    # --- Control Step ---
    # Calculate the control torque for the *current* state
    C_in = controller.calculate_control_torque(
        y=y_current,
        y_ref=y_ref,
        dt=DT,
        robot_params=robot
    )
    
    # --- Plant Step ---
    # We need a function for the solver that takes (t, y)
    # Use a lambda to "fix" the C_in argument for this time step
    dynamics_with_control = lambda t_solve, y_solve: robot.system_dynamics(
        t_solve, 
        y_solve, 
        C_in
    )
    
    # Simulate the *next* time step
    sol = solve_ivp(
        fun=dynamics_with_control,
        t_span=[t, t + DT],  # Simulate *only* this small interval
        y0=y_current,
        t_eval=[t + DT]     # We only want the state at the end
    )
    
    # --- Check if solver failed ---
    if sol.status != 0:
        print(f"WARNING: Solver failed at t={t} with status {sol.status}")
        print(f"Message: {sol.message}")
        break # Stop the simulation

    # --- FIX: AttributeError (Typo fix) ---
    # Force 'sol.y' (lowercase) to be a numpy array before flattening
    y_current = np.array(sol.y).flatten() # <--- TYPO WAS HERE (Y -> y)

print("Simulation complete.")

# --- 4. Process and Plot Results ---

# Extract data from history
time = history[:, 0]
theta_rad = history[:, 1]
theta_dot_rad_s = history[:, 2]
x_m = history[:, 3]
x_dot_m_s = history[:, 4]

# Convert angle to degrees for plotting (like the paper)
theta_deg = theta_rad * (180.0 / np.pi)

print("Plotting results...")

# Create two subplots, stacked vertically (like Figure 11)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Pitch Angle (theta)
ax1.plot(time, theta_deg, label=r'$\theta(t)$ (Angle)')
ax1.set_title('Replication of Figure 11: Stabilizing from 5 deg')
ax1.set_ylabel('Angle [degrees]')
ax1.grid(True)
ax1.legend()
ax1.set_ylim(-2, 6) # Set y-axis limits to match paper

# Plot 2: Position (x)
ax2.plot(time, x_m, label='x(t) (Position)', color='C1')
ax2.set_ylabel('Position [m]')
ax2.set_xlabel('Time [s]')
ax2.grid(True)
ax2.legend()
ax2.set_ylim(-0.02, 0.10) # Set y-axis limits to match paper

# Show the plot
plt.tight_layout()
plt.show()