import numpy as np
import pinocchio as pin
import os
import sys

# --- 1. PATH ROUTING ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "utils"))

from trajectory_math import generate_fourier_trajectory
from regressor_builder import build_observation_matrix
from base_params_calculator import get_base_parameter_count

# --- 2. CONFIGURATION ---
# Change this to "verification_coeffs.npy" to test your verification trajectory
COEFFS_FILENAME = "optimal_coeffs.npy" 

# Set the fundamental frequency based on the trajectory you are evaluating:
# - optimal_coeffs: 2 * np.pi * (2 / 65)
# - verification_coeffs: 2 * np.pi * (1 / 30)
WF_RAD = 2 * np.pi * (2 / 65)  

model_path = os.path.join(BASE_DIR, "robot_assets", "KR16_L6.urdf") 
coeffs_path = os.path.join(BASE_DIR, "data", "trajectories", COEFFS_FILENAME)

def main():
    if not os.path.exists(coeffs_path):
        raise FileNotFoundError(f"Cannot find coefficients at {coeffs_path}")

    print(f"Evaluating Trajectory: {COEFFS_FILENAME}")
    
    # Load the optimized parameters
    x = np.load(coeffs_path)
    
    # Automatically calculate the number of harmonics based on array length
    NUM_HARMONICS = int((len(x) - 6) / 12)
    print(f"Loaded {len(x)} coefficients. Identified {NUM_HARMONICS} harmonics.")

    model = pin.buildModelFromUrdf(model_path) 
    data = model.createData()
    
    BASE_PARAM_COUNT = get_base_parameter_count(model, data)

    # Recreate the time array (150 samples over one full cycle, matching the optimizer)
    t_array = np.linspace(0, 2 * np.pi / WF_RAD, num=150) 
    
    # Generate the kinematic trajectory
    q, dq, ddq = generate_fourier_trajectory(x, t_array, num_harmonics=NUM_HARMONICS, wf=WF_RAD)
    q = q - q[0] 

    # Build Regressor and calculate Condition Number
    Phi_M = build_observation_matrix(model, data, q, dq, ddq)
    
    col_norms = np.linalg.norm(Phi_M, axis=0)
    col_norms[col_norms == 0] = 1.0
    Phi_norm = Phi_M / col_norms
    
    s = np.linalg.svd(Phi_norm, compute_uv=False)
    idx = BASE_PARAM_COUNT - 1
    cond = s[0] / s[idx] if s[idx] > 1e-7 else 1e6

    # --- CALCULATE EXACT PENALTIES ---
    # 1. Boundary Penalty
    q_lower = np.array([-1.5, -0.5, -0.5, -2.0, -0.5, -2.0])
    q_upper = np.array([ 1.5,  0.5,  0.5,  2.0,  0.5,  2.0])
    lower_v = np.maximum(0, q_lower - q)
    upper_v = np.maximum(0, q - q_upper)
    boundary_penalty = 50000.0 * np.sum(lower_v**2 + upper_v**2)

    # 2. Velocity Reward (Negative Penalty)
    joint_weights = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0]) 
    velocity_reward = 0.1 * np.sum((dq**2) * joint_weights)

    # 3. Variance Penalty
    target_variances = np.array([0.2, 0.1, 0.1, 0.2, 0.2, 0.1])
    joint_variances = np.var(q, axis=0)
    variance_violations = np.maximum(0, target_variances - joint_variances)
    variance_penalty = 5000.0 * np.sum(variance_violations**2)

    # 4. Kinematic (Velocity/Acceleration Limits) Penalty
    dq_limit = np.deg2rad([156, 156, 156, 335, 355, 647])
    ddq_limit = 10.0
    vel_violations = np.maximum(0, np.abs(dq) - dq_limit)
    accel_violations = np.maximum(0, np.abs(ddq) - ddq_limit)
    kinematic_penalty = 50000.0 * np.sum(vel_violations**2 + accel_violations**2)

    # Final math
    total_cost = cond - velocity_reward + boundary_penalty + variance_penalty + kinematic_penalty

    # --- PRINT CLEAN RESULTS ---
    print("\n" + "="*50)
    print(f"--- EVALUATION: {COEFFS_FILENAME} ---")
    print("="*50)
    print(f"Condition Number (Base): {cond:.4f}")
    print(f"Velocity Reward:        -{velocity_reward:.4f} (Subtracts from cost)")
    print(f"Boundary Penalty:        {boundary_penalty:.4f}")
    print(f"Variance Penalty:        {variance_penalty:.4f}")
    print(f"Kinematic Penalty:       {kinematic_penalty:.4f}")
    print("-" * 50)
    print(f"TOTAL OBJECTIVE COST:    {total_cost:.4f}")
    print("="*50)

    if total_cost > 10000.0:
        print("\n⚠️ WARNING: This trajectory triggered massive safety penalties.")
    elif cond < 50.0:
        print("\n✅ SUCCESS: This is a highly optimal trajectory (Condition Number < 50).")

if __name__ == "__main__":
    main()