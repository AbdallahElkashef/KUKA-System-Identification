import numpy as np
import pandas as pd
import os

# ============================================================================
# CONFIGURATION: 6-DOF ROBOT JOINT SETTINGS
# ============================================================================
# Update the 'file', 'kt', and 'gear_ratio' for each joint.
# The first 3 links use SGMGV-13. 
# Kt = 1.78 Nm/A (From datasheet Pg 7 for 400V SGMGV-13D)
# IMPORTANT: You still need to input the GEAR_RATIO for your specific robot arm!

JOINTS_CONFIG = {
    "J1": {"file": "joint1_log.csv", "kt": 1.78, "gear_ratio": 125.0},
    "J2": {"file": "joint2_log.csv", "kt": 1.5, "gear_ratio": 125.0},
    "J3": {"file": "joint3_log.csv", "kt": 1.78, "gear_ratio": 125.0},
    "J4": {"file": "joint4_log.csv", "kt": 0.49, "gear_ratio": 74.444},  
    "J5": {"file": "joint5_log.csv", "kt": 0.49, "gear_ratio": 42.222},  
    "J6": {"file": "joint6_log.csv", "kt": 0.49, "gear_ratio": 24.117},  
}

COLUMN_NAME = "mV_Reading"  # The name of the column containing the mV data
# ============================================================================

def process_all_joints(config_dict, column_name):
    print(f"==================================================")
    print(f" STARTING SENSOR VARIANCE ANALYSIS")
    print(f"==================================================")

    final_variances = []

    for joint_name, params in config_dict.items():
        filepath = params["file"]
        kt = params["kt"]
        gear_ratio = params["gear_ratio"]
        
        print(f"\nProcessing {joint_name}...")
        print(f"  File: {filepath} | Kt: {kt} | Gear Ratio: {gear_ratio}")

        if not os.path.exists(filepath):
            print(f"  [!] ERROR: File '{filepath}' not found. Skipping...")
            final_variances.append(1.0) # Default back to LS mathematically if missing
            continue

        try:
            df = pd.read_csv(filepath)
            if column_name not in df.columns:
                print(f"  [!] ERROR: Column '{column_name}' not found. Skipping...")
                final_variances.append(1.0)
                continue
                
            raw_mV_data = df[column_name].values
            
            # 1. Convert mV to Amps (1mV/A from UNI-T datasheet)
            current_amps = raw_mV_data * 1.0  

            # 2. Convert Amps to Torque (Nm)
            torque_Nm = current_amps * kt * gear_ratio

            # 3. Calculate Variance
            variance = np.var(torque_Nm)
            final_variances.append(variance)
            
            print(f"  -> Variance: {variance:.4f} N^2m^2")

        except Exception as e:
            print(f"  [!] ERROR processing {joint_name}: {e}")
            final_variances.append(1.0)

    # --- Final Output Formatting ---
    print(f"\n==================================================")
    print(f" FINAL RESULTS FOR config.py")
    print(f"==================================================")
    
    formatted_array = ", ".join([f"{v:.4f}" for v in final_variances])
    print(f"\nTORQUE_VARIANCES = np.array([{formatted_array}])\n")
    print(f"==================================================")


if __name__ == "__main__":
    # --- For testing purposes, generate dummy CSVs if they don't exist ---
    for joint, params in JOINTS_CONFIG.items():
        if not os.path.exists(params["file"]):
            print(f"Generating dummy test data for {params['file']}...")
            # Simulate a noisy sensor reading around 5.0 mV
            dummy_data = np.random.normal(loc=5.0, scale=0.5, size=1000)
            pd.DataFrame({COLUMN_NAME: dummy_data}).to_csv(params["file"], index=False)
    
    # Run the batch processing script
    process_all_joints(JOINTS_CONFIG, COLUMN_NAME)