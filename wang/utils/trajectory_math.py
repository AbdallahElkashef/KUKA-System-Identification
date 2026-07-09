import numpy as np

def generate_fourier_trajectory(x, t_array, num_harmonics=5, wf=0.1):
    """
    x: array of 66 variables [offset_1, a_11, b_11... offset_6, a_65, b_65]
    Returns q, dq, ddq arrays for the given time sequence.
    """
    q = np.zeros((len(t_array), 6))
    dq = np.zeros((len(t_array), 6))
    ddq = np.zeros((len(t_array), 6))
    
    idx = 0
    for i in range(6): # For each joint
        q0 = x[idx]
        idx += 1
        
        q[:, i] = q0
        for k in range(1, num_harmonics + 1):
            a_k = x[idx]
            b_k = x[idx+1]
            idx += 2
            
            # Position
            q[:, i] += a_k * np.sin(k * wf * t_array) + b_k * np.cos(k * wf * t_array)
            # Velocity (Analytical Derivative)
            dq[:, i] += a_k * k * wf * np.cos(k * wf * t_array) - b_k * k * wf * np.sin(k * wf * t_array)
            # Acceleration (Analytical Derivative)
            ddq[:, i] += -a_k * (k * wf)**2 * np.sin(k * wf * t_array) - b_k * (k * wf)**2 * np.cos(k * wf * t_array)
            
    return q, dq, ddq