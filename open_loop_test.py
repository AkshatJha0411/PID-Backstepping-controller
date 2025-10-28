# test_open_loop.py
# Implements Phase 2, Step 2: The Open-Loop Test

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Import our RobotModel class from the other file in this folder
from robot_model import RobotModel

# --- 1. Setup ---

# Create an instance of our robot model.
# This object holds all the physical parameters (Mw, Mb, etc.).
robot = RobotModel()

# Define simulation parameters
T_START = 0.0   # Start time [s]
T_END = 5.0     # End time [s]
DT = 0.01       # Time step for evaluation [s]

# Define initial conditions (state vector y_0)
# y = [x1, x2, x3, x4] = [\theta, \dot{\theta}, x, \dot{x}]
theta_0 = 1.0 * (np.pi / 180.0)  # 1.0 degree initial tilt [rad]
theta_dot_0 = 0.0                # Initial angular velocity [rad/s]
x_0 = 0.0                        # Initial position [m]
x_dot_0 = 0.0                    # Initial velocity [m/s]

y0 = [theta_0, theta_dot_0, x_0, x_dot_0]

# --- 2. Run Open-Loop Simulation ---

# We need to simulate the system with ZERO control input.
# The `solve_ivp` function needs a callable that takes (t, y).
# Our `robot.system_dynamics` method takes (t, y, C_in).
# We use a lambda function to "wrap" our method, fixing C_in=0.0
open_loop_dynamics = lambda t, y: robot.system_dynamics(t, y, C_in=0.0)

# Define the time points where we want the solution
t_eval = np.arange(T_START, T_END, DT)

print(f"Running open-loop simulation for {T_END} seconds...")

# Run the ODE solver
# 'sol' will be an object containing the solution
sol = solve_ivp(
    fun=open_loop_dynamics,  # The function to integrate
    t_span=[T_START, T_END], # The time interval
    y0=y0,                   # The initial state
    t_eval=t_eval            # Time points to store the solution at
)

print("Simulation complete.")

# --- 3. Process and Plot Results ---

# The solution object 'sol' has two important attributes:
# sol.t: An array of the time points
# sol.y: An array where each *row* is a state variable.
#        sol.y[0] = x1 (theta)
#        sol.y[1] = x2 (theta_dot)
#        sol.y[2] = x3 (x)
#        sol.y[3] = x4 (x_dot)

# Extract states for plotting
time = sol.t
theta_rad = sol.y[0]
theta_deg = theta_rad * (180.0 / np.pi) # Convert to degrees for plotting

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time, theta_deg, label=r'$\theta(t)$ (angle)')
plt.title('Open-Loop Response (C_in = 0)')
plt.xlabel('Time [s]')
plt.ylabel('Pitch Angle [degrees]')
plt.grid(True)
plt.legend()
plt.show()

print("Plot displayed. Check if the angle increases, showing the 'fall'.")