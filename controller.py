# controller.py
# Implements Phase 2, Step 3: The Backstepping & PD Controllers

import numpy as np

class Controller:
    """
    Implements the PID Backstepping controller from Section V.
    This class holds the controller gains and calculates the
    necessary control torques.
    """
    def __init__(self):
        # --- Controller Gains (from Table IV) ---
        
        # Backstepping-Balance Controller
        self.k1 = 110.5
        self.k2 = 21.4
        self.c1 = 3.0
        
        # PD - Position Controller
        self.Kp_pos = 60.0
        self.Kd_pos = 7.5
        
        # PI - Rotation Controller (Not used in 2D sim, but here for completeness)
        self.Kp_rot = 47.5
        self.Ki_rot = 5.0
        
        # --- Controller State Variables ---
        # We need to store the integral of the angle error, z1
        self.z1 = 0.0  # z1 = \int e1 d\tau
    
    def reset_states(self):
        """ Resets any integral terms for a new simulation run. """
        self.z1 = 0.0

    def calculate_control_torque(self, y, y_ref, dt, robot_params):
        """
        Calculates the total control torque C_in = C_theta + C_x
        
        Args:
            y (list): The current state vector [x1, x2, x3, x4]
            y_ref (list): The reference state vector [x1_ref, x3_ref]
            dt (float): The time step [s], needed for integration
            robot_params (RobotModel): The robot parameters (L, Mb, etc.)
                                       needed for the control law.
        """
        
        # Unpack current state
        x1, x2, x3, x4 = y
        
        # Unpack reference state
        x1_ref, x3_ref = y_ref
        
        # Unpack necessary robot parameters
        L = robot_params.L
        Mb = robot_params.Mb
        
        # --- 1. Backstepping (Balance) Controller (Section V-B) ---
        
        # Reference pitch angle (x1_ref) is \theta_ref
        # For station-keeping and position control, \theta_ref = 0
        # For angle tracking (Scenario 3), it's non-zero.
        
        # x1_ref is the *commanded* pitch, which is 0 for pos control.
        # x1_ref_dot and x1_ref_ddot are also 0.
        x1_ref_dot = 0.0
        x1_ref_ddot = 0.0 # \ddot{x}_{1ref} in Eq (31)

        # Define tracking error e1
        e1 = x1_ref - x1
        
        # Update integral of e1 (z1)
        # z1 = \int e1 d\tau
        self.z1 += e1 * dt
        
        # Define virtual control \alpha
        alpha = self.k1 * e1 + self.c1 * self.z1 + x1_ref_dot
        
        # Define error e2
        e2 = alpha - x2
        
        # We need f1(x1) and f2(x1, x2) from the model
        # We can't get them directly from robot.system_dynamics,
        # so we must re-calculate them here. This is a common
        # part of model-based control like backstepping.
        
        # --- Re-calculate f1, f2, g1 from the model ---
        # Note: This is a necessary duplication of logic from robot_model.py
        # because the controller *needs to know the model*.
        sin_x1 = np.sin(x1)
        cos_x1 = np.cos(x1)
        
        # Parameters needed from the robot model
        Mw, R, g = robot_params.Mw, robot_params.R, robot_params.g

        DENOM_1 = (0.75 * (Mw * R + Mb * L * cos_x1) * cos_x1 / ((2 * Mw + Mb) * L)) - 1
        
        f1 = (-0.75 * g * sin_x1 / L) / DENOM_1
        f2 = (0.75 * Mb * L * sin_x1 * cos_x1 * (x2**2) / ((2 * Mw + Mb) * L)) / DENOM_1
        
        g1_num = (0.75 * (1 + sin_x1**2) / (Mb * L**2)) + (0.75 * cos_x1 / ((2 * Mw + Mb) * R * L))
        g1 = g1_num / DENOM_1
        # --- End of re-calculated terms ---

        # Calculate C_theta using Equation (31)
        C_theta_num = (1 + self.c1 - self.k1**2) * e1 + (self.k1 + self.k2) * e2 - self.k1 * self.c1 * self.z1 + x1_ref_ddot - f1 - f2
        C_theta = C_theta_num / g1
        
        
        # --- 2. PD (Position) Controller (Section V-C) ---
        
        # Reference velocity is 0
        x4_ref = 0.0
        
        # Calculate errors
        e_pos = x3_ref - x3
        e_vel = x4_ref - x4
        
        # PD control law
        # C_x = self.Kp_pos * e_pos + self.Kd_pos * e_vel
        
        # PD control law
        # The sign is inverted to correctly interact with the backstepping
        # controller. A positive error (x_ref > x) must create a
        # negative torque to "pull" the robot forward via tilting.
        C_x = - (self.Kp_pos * e_pos + self.Kd_pos * e_vel) # <--- FIXED SIGN HERE
        
        # --- 3. Total Control Torque ---
        # As per Figure 7, the total torque for the 2D case is C_theta + C_x
        C_in = C_theta + C_x
        
        return C_in