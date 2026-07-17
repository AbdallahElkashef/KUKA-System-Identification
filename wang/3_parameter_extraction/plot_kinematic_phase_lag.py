import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- 1. UNIVERSAL PATH ROUTING ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "utils"))

from trajectory_math import generate_fourier_trajectory

dataset_path = os.path.join(BASE_DIR, "data", "training_datasets", "identification_dataset.npz")
coeffs_path = os.path.join(BASE_DIR, "data", "trajectories", "optimal_coeffs.npy")
output_plot_path = os.path.join(BASE_DIR, "data", "real_phase_lag_slide.png")

# Trajectory Constants (From your optimization script)
WF_RAD = 2 * np.pi * (2 / 65)
NUM_HARMONICS = 5
JOINT_TO_PLOT = 0  # Joint 1

def main():
    if not os.path.exists(dataset_path) or not os.path.exists(coeffs_path):
        raise FileNotFoundError("Could not find the dataset or the coefficients.")

    print("Loading actual MuJoCo data and Fourier coefficients...")
    data_in = np.load(dataset_path)
    t = data_in['t']
    q_meas_all = data_in['q']
    
    coeffs = np.load(coeffs_path)

    print("Reconstructing mathematical commanded trajectory...")
    q_cmd_all, _, _ = generate_fourier_trajectory(coeffs, t, num_harmonics=NUM_HARMONICS, wf=WF_RAD)
    q_cmd_all = q_cmd_all - q_cmd_all[0]

    q_cmd = q_cmd_all[:, JOINT_TO_PLOT]
    q_meas = q_meas_all[:, JOINT_TO_PLOT]

    # --- 2. PRESENTATION-READY FIGURE SETUP ---
    # FIXED: Removed dpi=300 from here to prevent Windows display scaling issues
    fig, ax = plt.subplots(figsize=(12, 6.75))
    
    ax.plot(t, q_cmd, color='#0033a0', label='Commanded Math Trajectory ($q_d$)', linewidth=2.5, linestyle='--')
    ax.plot(t, q_meas, color='#c8102e', label='Measured MuJoCo State ($q_m$)', linewidth=2.5)

    ax.set_title(f"Discovery of Kinematic Tracking Error (Joint {JOINT_TO_PLOT + 1})", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Joint Position (rad)", fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, 10)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
    ax.legend(loc='lower right', fontsize=12, framealpha=1.0, edgecolor='black')
    ax.tick_params(axis='both', which='major', labelsize=12)

    # --- 3. AUTO-LOCATE PEAK AND CREATE ZOOMED INSET ---
    print("Auto-locating tracking error peak...")
    search_mask = (t > 1.0) & (t < 8.0)
    t_search = t[search_mask]
    q_search = q_meas[search_mask]
    
    peak_idx = np.argmax(np.abs(q_search))
    peak_time = t_search[peak_idx]

    axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
    
    axins.plot(t, q_cmd, color='#0033a0', linewidth=3, linestyle='--')
    axins.plot(t, q_meas, color='#c8102e', linewidth=3)
    
    # FIXED: Shrunk the zoom window to 0.1s to make the tiny phase lag gap highly visible
    zoom_window = 0.1 
    axins.set_xlim(peak_time - zoom_window, peak_time + zoom_window)
    
    y_min_zoom = min(np.min(q_cmd[(t > peak_time - zoom_window) & (t < peak_time + zoom_window)]), 
                     np.min(q_meas[(t > peak_time - zoom_window) & (t < peak_time + zoom_window)]))
    y_max_zoom = max(np.max(q_cmd[(t > peak_time - zoom_window) & (t < peak_time + zoom_window)]), 
                     np.max(q_meas[(t > peak_time - zoom_window) & (t < peak_time + zoom_window)]))
    
    padding = (y_max_zoom - y_min_zoom) * 0.15
    axins.set_ylim(y_min_zoom - padding, y_max_zoom + padding)
    axins.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    
    ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=2.0, alpha=0.6)

    # --- 4. SAVE OUT ---
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    
    # FIXED: Placed dpi=300 directly in the savefig command so only the file gets upscaled
    plt.savefig(output_plot_path, format='png', dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: Real data presentation graphic saved to: {output_plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()