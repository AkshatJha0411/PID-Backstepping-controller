# robot_model.py
# Implements Phase 2, Step 1: The System Model (Plant)
# Refactored to an Object-Oriented (OOP) style.

import numpy as np

class RobotModel:
    """
    Holds the physical parameters and implements the system dynamics
    for the two-wheeled self-balancing robot.
    """
    def __init__(self):
        # --- Physical Parameters (from Table III) ---
        
        # Mass of the wheel [kg]
        self.Mw = 0.5
        # Mass of the body [kg]
        self.Mb = 7.0
        # Radius of the wheel [m]
        self.R = 0.07
        # Distance from wheel axis to body's center of gravity [m]
        self.L = 0.3
        # Gravity constant [m/s^2]
        self.g = 9.8

    def system_dynamics(self, t, y, C_in):
        """
        Calculates the derivatives of the state vector.
        This method will be passed to the ODE solver.

        The state vector 'y' is:
        y = [x1, x2, x3, x4]
        x1 = theta      (pitch angle, rad)
        x2 = theta_dot (pitch angular velocity, rad/s)
        x3 = x          (position, m)
        x4 = x_dot      (velocity, m/s)

        The input 'C_in' is the total torque from the controllers:
        C_in = C_theta + C_x
        
        This implementation is based on the state equations (19.1) to (19.4)
        [cite_start]and their corresponding function definitions [cite: 267-305].
        """
        
        # Unpack state vector
        x1, x2, x3, x4 = y
        
        # Pre-calculate trig functions for efficiency
        sin_x1 = np.sin(x1)
        cos_x1 = np.cos(x1)
        
        # --- Calculate \dot{x}_2 (Pitch Acceleration \ddot{\theta}) ---
        # This is based on Equation (19.2): \dot{x}_2 = f_1 + f_2 + g_1 * C_in
        
        # Common denominator for f_1, f_2, g_1
        DENOM_1 = (0.75 * (self.Mw * self.R + self.Mb * self.L * cos_x1) * cos_x1 / ((2 * self.Mw + self.Mb) * self.L)) - 1
        
        # f_1(x_1)
        f1 = (-0.75 * self.g * sin_x1 / self.L) / DENOM_1
        
        # f_2(x_1, x_2)
        f2 = (0.75 * self.Mb * self.L * sin_x1 * cos_x1 * (x2**2) / ((2 * self.Mw + self.Mb) * self.L)) / DENOM_1
        
        # g_1(x_1)
        g1_num = (0.75 * (1 + sin_x1**2) / (self.Mb * self.L**2)) + (0.75 * cos_x1 / ((2 * self.Mw + self.Mb) * self.R * self.L))
        g1 = g1_num / DENOM_1
        
        
        # --- Calculate \dot{x}_4 (Linear Acceleration \ddot{x}) ---
        # This is based on Equation (19.4): \dot{x}_4 = f_3 + f_4 + g_2 * C_in
        
        # Common denominator for f_3, f_4, g_2
        DENOM_2 = (2 * self.Mw + self.Mb) - (0.75 * (self.Mw * self.R + self.Mb * self.L * cos_x1) * cos_x1 / self.L)
        
        # f_3(x_1)
        f3 = (-0.75 * self.g * (self.Mw * self.R + self.Mb * self.L * cos_x1) * sin_x1 / self.L) / DENOM_2
        
        # f_4(x_1, x_2)
        f4 = (self.Mb * self.L * sin_x1 * (x2**2)) / DENOM_2
        
        # g_2(x_1)
        g2_num = (0.75 * (self.Mw * self.R + self.Mb * self.L * cos_x1) * (1 + sin_x1**2) / (self.Mb * self.L**2)) + (1 / self.R)
        g2 = g2_num / DENOM_2

        
        # --- Assemble the State Derivatives ---
        
        # \dot{x}_1 = x_2
        x1_dot = x2
        
        # \dot{x}_2 = ...
        x2_dot = f1 + f2 + g1 * C_in
        
        # \dot{x}_3 = x_4
        x3_dot = x4
        
        # \dot{x}_4 = ...
        x4_dot = f3 + f4 + g2 * C_in
        
        # Return the derivative vector as a numpy array
        return np.array([x1_dot, x2_dot, x3_dot, x4_dot])