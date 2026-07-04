import math

import numpy as np

# Define DH parameters
d1, d2, d3, d4, d5, d6 = 0.675, 0, 0, 0.970, 0, 0.115
a1, a2, a3, a4, a5, a6 = 0.26, 0.680, 0.035, 0, 0, 0
alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = -np.pi/2, 0, -np.pi/2, np.pi/2, -np.pi/2, 0

# Standard DH Transformation Function
def DH_TF(q, d, a, alpha):
    return np.array([
        [np.cos(q), -np.sin(q) * np.cos(alpha), np.sin(q) * np.sin(alpha), a * np.cos(q)],
        [np.sin(q), np.cos(q) * np.cos(alpha), -np.cos(q) * np.sin(alpha), a * np.sin(q)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(q_values):
    q1, q2, q3, q4, q5, q6 = q_values  
    
    # Compute individual transformation matrices
    T0_1 = DH_TF(q1, d1, a1, alpha1)
    T1_2 = DH_TF(q2 - np.pi/2, d2, a2, alpha2) 
    T2_3 = DH_TF(q3, d3, a3, alpha3)
    T3_4 = DH_TF(q4, d4, a4, alpha4)
    T4_5 = DH_TF(q5, d5, a5, alpha5)
    T5_6 = DH_TF(q6, d6, a6, alpha6)
    
    # Final 4x4 End-Effector Transformation Matrix
    T0_EE = T0_1 @ T1_2 @ T2_3 @ T3_4 @ T4_5 @ T5_6
    
    # 1. Extract Position coordinates
    x = T0_EE[0, 3]
    y = T0_EE[1, 3]
    z = T0_EE[2, 3]
    
    # 2. Calculate Radial Distance (Distance from Base)
    r = math.hypot(x, y)
    # Note: np.linalg.norm([x, y, z]) also works perfectly here.
    
    return x, y, z, r
