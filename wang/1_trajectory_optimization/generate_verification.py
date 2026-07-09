import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
import os
import sys

# --- UPDATED ROUTING ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "utils"))

from trajectory_math import generate_fourier_trajectory
from base_params_calculator import get_base_parameter_count

model_path = os.path.join(BASE_DIR, "robot_assets", "KR16_L6.urdf") 
output_path = os.path.join(BASE_DIR, "data", "trajectories", "verification_coeffs.npy")

model = pin.buildModelFromUrdf(model_path) 
data = model.createData()

BASE_PARAM_COUNT = get_base_parameter_count(model, data)
WF_RAD = 2 * np.pi * (1 / 30) 
NUM_HARMONICS = 5

def objective_function(x):
    t_array = np.linspace(0, 2 * np.pi / WF_RAD, num=150) 
    
    q, dq, ddq = generate_fourier_trajectory(x, t_array, num_harmonics=NUM_HARMONICS, wf=WF_RAD)
    q = q - q[0] 
    
    Phi_M = []
    for i in range(len(t_array)):
        phi_inertial = pin.computeJointTorqueRegressor(model, data, q[i], dq[i], ddq[i])
        phi_fric = np.zeros((6, 12))
        for j in range(6):
            phi_fric[j, 2*j] = dq[i, j]               
            phi_fric[j, 2*j + 1] = np.sign(dq[i, j])  
        phi_total = np.hstack((phi_inertial, phi_fric))
        Phi_M.append(phi_total)
        
    Phi_M = np.vstack(Phi_M)
    col_norms = np.linalg.norm(Phi_M, axis=0)
    col_norms[col_norms == 0] = 1.0
    Phi_norm = Phi_M / col_norms
    
    s = np.linalg.svd(Phi_norm, compute_uv=False)
    idx = BASE_PARAM_COUNT - 1
    cond = s[0] / s[idx] if s[idx] > 1e-7 else 1e6
    
    # --- FIXED: SAFE HEMISPHERE BOUNDARIES ---
    q_lower = np.array([-1.5, -0.5, -0.5, -2.0, -0.5, -2.0])
    q_upper = np.array([ 1.5,  0.5,  0.5,  2.0,  0.5,  2.0])
    lower_v = np.maximum(0, q_lower - q)
    upper_v = np.maximum(0, q - q_upper)
    boundary_penalty = 50000.0 * np.sum(lower_v**2 + upper_v**2)
    
    joint_weights = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0]) 
    velocity_reward = 0.1 * np.sum((dq**2) * joint_weights)
    
    target_variances = np.array([0.2, 0.1, 0.1, 0.2, 0.2, 0.1])
    joint_variances = np.var(q, axis=0)
    variance_violations = np.maximum(0, target_variances - joint_variances)
    variance_penalty = 5000.0 * np.sum(variance_violations**2)

    dq_limit = np.deg2rad([156, 156, 156, 335, 355, 647])
    ddq_limit = 10.0
    
    vel_violations = np.maximum(0, np.abs(dq) - dq_limit)
    accel_violations = np.maximum(0, np.abs(ddq) - ddq_limit)
    kinematic_penalty = 50000.0 * np.sum(vel_violations**2 + accel_violations**2)
    
    total_cost = cond - velocity_reward + boundary_penalty + variance_penalty + kinematic_penalty
    print(f"  -> Cond: {cond:.1f} | Vel Rew: {velocity_reward:.1f} | Bound Pen: {boundary_penalty:.1f} | Var Pen: {variance_penalty:.1f} | Kin Pen: {kinematic_penalty:.1f}")
    return total_cost

if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path): os.remove(output_path)
    
    num_coeffs = 6 * 2 * NUM_HARMONICS + 6
    best_cost, best_x = float('inf'), None
    
    for trial in range(200): 
        x0 = np.random.uniform(-0.5, 0.5, num_coeffs) 
        res = minimize(objective_function, x0, method='SLSQP', options={'maxiter': 3000, 'ftol': 1e-4}) 
        print(f"Trial {trial} finished. Success: {res.success} | Message: {res.message}")
        if res.fun < best_cost:
            best_cost = res.fun
            best_x = res.x
            
    print(f"\n--- SWEEP COMPLETE ---")
    if best_x is not None:
        if best_cost < 10000.0: 
            np.save(output_path, best_x)
            print(f"SUCCESS: Saved verification coefficients. Final Cost: {best_cost:.2f}")
        else:
            print(f"PARTIAL FAILURE: Found a trajectory, but it hits a safety boundary. Cost: {best_cost:.2f}")
            np.save(output_path, best_x) 
    else:
        print("CRITICAL FAILURE: Solver completely locked up on all 50 trials.")