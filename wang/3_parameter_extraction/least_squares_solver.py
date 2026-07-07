import numpy as np
import pinocchio as pin
import os
from scipy.signal import butter, filtfilt, savgol_filter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "data", "training_datasets", "identification_dataset.npz")
model_path = os.path.join(BASE_DIR, "robot_assets", "KR16_L6.urdf")
pi_hat_path = os.path.join(BASE_DIR, "data", "training_datasets", "pi_hat.npy")

data_in = np.load(dataset_path)
t, q, dq, tau = data_in['t'], data_in['q'], data_in['dq'], data_in['tau']

model = pin.buildModelFromUrdf(model_path)
pdata = model.createData()

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

print("Constructing the Observation Matrix (W) with Friction...")
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

print("Executing Safe Normalized Ordinary Least Squares...")

# --- FIXED: SAFE NORMALIZATION ---
scale = np.linalg.norm(W, axis=0)

# If a parameter is unobservable/noise, its scale is tiny.
# By locking small scales to 1.0, we completely prevent the Scaling Explosion 
# when we divide by scale at the end.
scale[scale < 1e-2] = 1.0  

W_norm = W / scale

# Use standard pinv on the safely normalized matrix
W_norm_pinv = np.linalg.pinv(W_norm, rcond=1e-3) 
pi_scaled = W_norm_pinv @ tau_flat

# Scale back to physical units (unobservable parameters stay safely near 0)
pi_hat = pi_scaled / scale

print(f"Successfully identified {len(pi_hat)} parameters (including friction).")

os.makedirs(os.path.dirname(pi_hat_path), exist_ok=True)
np.save(pi_hat_path, pi_hat)
print(f"Parameters saved to: {pi_hat_path}")