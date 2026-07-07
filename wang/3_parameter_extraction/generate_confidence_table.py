import numpy as np
import pinocchio as pin
import os
from scipy.signal import butter, filtfilt, savgol_filter

# --- 1. UNIVERSAL PATH ROUTING ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "data", "training_datasets", "identification_dataset.npz")
model_path = os.path.join(BASE_DIR, "robot_assets", "KR16_L6.urdf")
pi_hat_path = os.path.join(BASE_DIR, "data", "training_datasets", "pi_hat.npy")

def main():
    print("Loading data and physical models...")
    data_in = np.load(dataset_path)
    t, q, dq, tau = data_in['t'], data_in['q'], data_in['dq'], data_in['tau']
    pi_hat = np.load(pi_hat_path)

    model = pin.buildModelFromUrdf(model_path)
    pdata = model.createData()

    # --- 2. DATA FILTERING ---
    dt = t[1] - t[0]
    ddq_raw = np.gradient(dq, dt, axis=0)
    
    b, a = butter(4, 0.1, btype='low') 
    ddq_filt = filtfilt(b, a, ddq_raw, axis=0)
    dq_filt = filtfilt(b, a, dq, axis=0) 
    tau_filt = savgol_filter(tau, window_length=51, polyorder=2, axis=0)

    trim = int(0.5 / dt)
    t = t[trim:-trim]
    q = q[trim:-trim]
    dq_filt = dq_filt[trim:-trim]
    ddq_filt = ddq_filt[trim:-trim]
    tau_filt = tau_filt[trim:-trim]

    # --- 3. BUILD REGRESSOR MATRIX ---
    print("Reconstructing the Observation Matrix (W) for statistical analysis...")
    W = []
    tau_flat = []
    for i in range(len(t)):
        phi_inertial = pin.computeJointTorqueRegressor(model, pdata, q[i], dq_filt[i], ddq_filt[i])
        phi_fric = np.zeros((6, 12))
        for j in range(6):
            phi_fric[j, 2*j] = dq_filt[i, j]               
            phi_fric[j, 2*j + 1] = np.sign(dq_filt[i, j])  
            
        phi_total = np.hstack((phi_inertial, phi_fric))
        W.append(phi_total)
        tau_flat.append(tau_filt[i])
        
    W = np.vstack(W)          
    tau_flat = np.concatenate(tau_flat) 

    # --- 4. STATISTICAL CONFIDENCE MATH (Equations 7-10) ---
    print("Calculating Covariance and Relative Standard Deviations...")
    tau_predicted = W @ pi_hat
    residuals = tau_flat - tau_predicted

    Nn = len(tau_flat)  
    m = len(pi_hat)     
    sigma_sq_error = np.sum(residuals**2) / (Nn - m)

    W_pinv_unscaled = np.linalg.pinv(W, rcond=1e-3)
    cov_matrix = sigma_sq_error * (W_pinv_unscaled @ W_pinv_unscaled.T)

    sigma_beta = np.sqrt(np.abs(np.diag(cov_matrix)))
    sigma_beta_rel_percent = 100 * (sigma_beta / (np.abs(pi_hat) + 1e-10))

    # --- 5. SYMBOLIC MAPPING FOR THESIS TABLE ---
    # Pinocchio's standard spatial inertia order + our 2 custom friction parameters
    param_names = []
    for i in range(1, 7):
        param_names.extend([
            f"M_{i}", f"MX_{i}", f"MY_{i}", f"MZ_{i}",
            f"Ixx_{i}", f"Ixy_{i}", f"Iyy_{i}", f"Ixz_{i}", f"Iyz_{i}", f"Izz_{i}",
            f"Fv_{i}", f"Fc_{i}"
        ])

    # --- 6. PRINT FORMATTED TABLE ---
    print("\n" + "="*55)
    print(f"{'Parameter Symbol':<20} | {'Beta (Value)':<15} | {'Std Dev (%)':<15}")
    print("-" * 55)
    for i in range(m):
        # Only print parameters that the SVD algorithm actually identified (non-zero)
        if np.abs(pi_hat[i]) > 1e-5:
            print(f"{param_names[i]:<20} | {pi_hat[i]:>12.4f}    | {sigma_beta_rel_percent[i]:>10.4f}%")
    print("="*55)
    print("NOTE: Parameters not listed were mathematically eliminated by the solver as unidentifiable noise.")

if __name__ == "__main__":
    main()