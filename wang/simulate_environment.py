import os
import time
import numpy as np
import mujoco
import mujoco.viewer

from param_identification.trajectory_math import generate_fourier_trajectory

# --- 1. SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "robot_assets", "KR16_L6.xml")
coeffs_path = os.path.join(BASE_DIR, "data", "trajectories", "optimal_coeffs.npy")
dataset_out_path = os.path.join(BASE_DIR, "data", "training_datasets", "identification_dataset.npz")                   

# --- 2. TRAJECTORY PARAMETERS ---
WF_RAD = 2 * np.pi * (2 / 65)  
CYCLES = 8  # Set to 8 to match the time-domain averaging from the research paper

def main():
    if not os.path.exists(coeffs_path):
        raise FileNotFoundError(f"Cannot find optimal coefficients at {coeffs_path}")
    
    x_opt = np.load(coeffs_path)
    
    # --- DYNAMIC HARMONIC DETECTION ---
    NUM_HARMONICS = int((len(x_opt) - 6) / 12)
    print(f"Loaded {len(x_opt)} coefficients. Automatically set NUM_HARMONICS to {NUM_HARMONICS}.")
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # --- 3. THE REALISTIC FIX: HIGH PEDESTAL ---
    # Disable default XML actuators so they don't fight our custom PD controller
    if model.nu > 0:
        model.actuator_ctrlrange[:] = 0
        model.actuator_forcerange[:] = 0
    
    # --- 4. PRECOMPUTE TRAJECTORY ---
    T_cycle = 2 * np.pi / WF_RAD
    sim_duration = T_cycle * CYCLES
    dt = model.opt.timestep  
    num_steps = int(sim_duration / dt)
    t_array = np.linspace(0, sim_duration, num_steps)
    
    print("Precomputing trajectory...")
    q_des, dq_des, ddq_des = generate_fourier_trajectory(x_opt, t_array, num_harmonics=NUM_HARMONICS, wf=WF_RAD)
    
    # --- THE MATHEMATICAL ANCHOR ---
    # Mathematically force the trajectory to start exactly at the native [0,0,0,0,0,0] L-shape
    q_des = q_des - q_des[0]
    
    # --- 5. CONTROL GAINS FOR KUKA KR 16 L6 ---
    # Scaled to match the actual mass of each link in the URDF.
    Kp = np.array([20000, 20000, 20000, 2500, 800, 10])
    Kd = np.array([1000, 1000, 1000, 150, 50, 1])
    
    # --- 6. DATA ARRAYS ---
    recorded_q = np.zeros((num_steps, model.nq))
    recorded_dq = np.zeros((num_steps, model.nv))
    recorded_tau = np.zeros((num_steps, model.nv))
    
    # --- 7. INITIALIZE ROBOT ---
    data.qpos[:] = q_des[0]
    data.qvel[:] = dq_des[0]
    mujoco.mj_forward(model, data) 
    
    # Dramatically reduce the torque limits for the lightweight wrist joints
    tau_limit = np.array([3000, 3000, 3000, 100, 50, 5])
    
    # --- 8. SIMULATION LOOP ---
    print("Launching MuJoCo Viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # --- SET DEFAULT CAMERA VIEW ---
        viewer.cam.distance = 5.0          
        viewer.cam.elevation = -20         
        viewer.cam.azimuth = 135           
        viewer.cam.lookat[:] = [0, 0, 2.5] 

        for i in range(num_steps):
            if not viewer.is_running():
                print("Viewer closed early.")
                break
            
            # Read current state
            q_curr = data.qpos[:]
            dq_curr = data.qvel[:]
            
            # Read Feedforward (Gravity & Coriolis) from the previous step
            tau_ff = data.qfrc_bias[:]
            
            # Calculate Errors
            error = q_des[i] - q_curr
            d_error = dq_des[i] - dq_curr
            
            # Calculate PD Control + Feedforward
            tau_pd = (Kp * error) + (Kd * d_error)
            tau_applied = tau_pd + tau_ff
            
            # Clip torques
            tau_applied = np.clip(tau_applied, -tau_limit, tau_limit)
            
            # Apply torques and step physics
            data.qfrc_applied[:] = tau_applied
            mujoco.mj_step(model, data)
            
            # Log data
            recorded_q[i, :] = data.qpos[:]
            recorded_dq[i, :] = data.qvel[:]
            recorded_tau[i, :] = tau_applied
            
            # Render at 60 FPS
            if i % int(1 / (60 * dt)) == 0:
                viewer.sync()
                
                # SPEED CONTROL
                time.sleep(1 / 120.0)
                
    # --- 9. SAVE DATASET ---
    os.makedirs(os.path.dirname(dataset_out_path), exist_ok=True)
    np.savez(dataset_out_path, t=t_array, q=recorded_q, dq=recorded_dq, tau=recorded_tau)
    print(f"Simulation complete! Clean data saved to: {dataset_out_path}")

if __name__ == "__main__":
    main()