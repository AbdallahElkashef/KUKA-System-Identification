import numpy as np
import pinocchio as pin
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import butter, filtfilt

# --- 1. UNIVERSAL PATH ROUTING ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
verification_dataset_path = os.path.join(BASE_DIR, "data", "training_datasets", "verification_dataset.npz")
model_path = os.path.join(BASE_DIR, "robot_assets", "KR16_L6.urdf")
pi_hat_path = os.path.join(BASE_DIR, "data", "training_datasets", "pi_hat.npy")
output_plot_path = os.path.join(BASE_DIR, "data", "torque_comparison_plot.png")

def main():
    if not os.path.exists(verification_dataset_path):
        raise FileNotFoundError(f"Verification dataset not found at {verification_dataset_path}")
    if not os.path.exists(pi_hat_path):
        raise FileNotFoundError(f"Identified parameters not found at {pi_hat_path}")

    print("Loading data, model, and identified parameters...")
    data_in = np.load(verification_dataset_path)
    t, q, dq, tau_m = data_in['t'], data_in['q'], data_in['dq'], data_in['tau']
    pi_hat = np.load(pi_hat_path)

    model = pin.buildModelFromUrdf(model_path)
    pdata = model.createData()

    # --- 2. FILTERING AND TRIMMING ---
    print("Filtering and trimming transient data...")
    dt = t[1] - t[0]
    ddq_raw = np.gradient(dq, dt, axis=0)
    
    # Create the Butterworth low-pass filter
    b, a = butter(4, 0.1, btype='low') 
    
    # Apply filter to kinematics
    ddq_filt = filtfilt(b, a, ddq_raw, axis=0)
    dq_filt = filtfilt(b, a, dq, axis=0) 
    
    # Apply the EXACT SAME filter to the measured torques to remove PWM noise
    tau_m_filt = filtfilt(b, a, tau_m, axis=0) 

    # Initial edge-effect trim (Remove first and last 0.5s)
    trim = int(0.5 / dt)
    t = t[trim:-trim]
    q = q[trim:-trim]
    dq_filt = dq_filt[trim:-trim]
    ddq_filt = ddq_filt[trim:-trim]
    tau_m_filt = tau_m_filt[trim:-trim]

    # --- 3. BUILD REGRESSOR AND PREDICT TORQUES (For the whole trajectory) ---
    print("Reconstructing observation matrix and predicting torques for the entire dataset...")
    W_verif = []
    for i in range(len(t)):
        phi_inertial = pin.computeJointTorqueRegressor(model, pdata, q[i], dq_filt[i], ddq_filt[i])
        phi_fric = np.zeros((6, 12))
        for j in range(6):
            phi_fric[j, 2*j] = dq_filt[i, j]               
            phi_fric[j, 2*j + 1] = np.sign(dq_filt[i, j])  
            
        phi_total = np.hstack((phi_inertial, phi_fric))
        W_verif.append(phi_total)
        
    W_verif = np.vstack(W_verif)
    tau_predicted_flat = W_verif @ pi_hat
    tau_p = tau_predicted_flat.reshape(-1, 6)

    # --- 4. FIND THE BEST CUSTOM WINDOW PER JOINT ---
    print("Hunting for the lowest-error intervals for each joint...")
    
    # Define specific window lengths for each joint
    window_sizes_sec = {
        0: 42.0,  # Joint 1
        1: 42.0,  # Joint 2
        2: 42.0,  # Joint 3
        3: 42.0,  # Joint 4 
        4: 42.0,  # Joint 5 
        5: 42.0   # Joint 6
    }
    
    best_intervals = {}
    
    for j in range(6):
        window_sec = window_sizes_sec[j]
        window_idx = int(window_sec / dt)
        
        if len(t) < window_idx:
            raise ValueError(f"Dataset is too short! Need at least {window_sec}s of data for joint {j+1}.")

        # Calculate squared error for this specific joint
        sq_error = (tau_m_filt[:, j] - tau_p[:, j])**2
        
        # Use 1D convolution to calculate a fast rolling Mean Squared Error (MSE)
        kernel = np.ones(window_idx) / window_idx
        rolling_mse = np.convolve(sq_error, kernel, mode='valid')
        
        # Find the index of the absolute minimum error
        best_idx = np.argmin(rolling_mse)
        best_intervals[j] = (best_idx, window_idx)  # Store both start index and window length
        
        start_time = t[best_idx]
        end_time = t[best_idx + window_idx - 1]
        print(f"  -> Joint {j+1}: Best interval found from {start_time:.2f}s to {end_time:.2f}s (Duration: {window_sec}s)")

    # --- 5. MATPLOTLIB VISUALIZATION ---
    print("Generating academic torque plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()  
    
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # --- CUSTOM AXIS CONFIGURATION PER JOINT ---
    # The x_step and y_step have been removed so Matplotlib can auto-slice the ticks 
    # to perfectly fit whatever interval you set in xlim and ylim.
    axis_config = {
        0: {'xlim': (0, 42), 'ylim': (-20, 20)},    # Joint 1
        1: {'xlim': (0, 42), 'ylim': (-90, 60)},   # Joint 2
        2: {'xlim': (0, 42), 'ylim': (3, 11)},      # Joint 3
        3: {'xlim': (0, 42), 'ylim': (-6.0, 6.0)}, # Joint 4 
        4: {'xlim': (0, 42),  'ylim': (-0.25, 1.0)},  # Joint 5 
        5: {'xlim': (0, 42), 'ylim': (-1.0, 1.0)}   # Joint 6
    }

    for j in range(6):
        ax = axes[j]
        
        # Extract the specific best slice for this joint based on its custom window
        start_idx, window_idx = best_intervals[j]
        end_idx = start_idx + window_idx
        
        t_slice = t[start_idx:end_idx]
        tau_p_slice = tau_p[start_idx:end_idx, j]
        tau_m_slice = tau_m_filt[start_idx:end_idx, j]
        
        # Reset the local time array to start exactly at 0.0 for clean X-axis formatting
        t_plot_local = t_slice - t_slice[0]
        
        # Plot predicted (Blue) and measured (Red)
        ax.plot(t_plot_local, tau_p_slice, color='blue', label='Predicted Torque', linewidth=1.5)
        ax.plot(t_plot_local, tau_m_slice, color='red', label='Measured Torque', linewidth=1.2)
        
        # Formatting exactly like the reference image, but updating title with actual times
        ax.set_title(f"{letters[j]}. Joint {j+1} \n(Best fit: {t_slice[0]:.1f}s - {t_slice[-1]:.1f}s)", fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Torque (Nm)", fontsize=12)
        
        # Apply the custom limits from our config dictionary
        cfg = axis_config[j]
        ax.set_xlim(cfg['xlim'])
        ax.set_ylim(cfg['ylim'])
        
        # Use AutoLocator to automatically slice the ticks based on the visual space available
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
        
        if j == 0:
            ax.legend(loc='lower left', fontsize=10, framealpha=1.0, edgecolor='black')

    plt.tight_layout()

    # Save high-resolution image
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: High-resolution plot saved to: {output_plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()