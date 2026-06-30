import pandas as pd
import numpy as np

# Load the trajectory data
df = pd.read_csv('trajectory_pso.csv')

# Define conversion factors
rad_to_deg = 180.0 / np.pi
rad_s_to_rpm = 60.0 / (2 * np.pi)
rad_s2_to_rpm2 = 3600.0 / (2 * np.pi) # Revolutions per minute squared

# If you meant RPM/sec for acceleration, use this instead:
# rad_s2_to_rpm_s = 60.0 / (2 * np.pi) 

# Apply conversions based on the column names
for col in df.columns:
    if col.startswith('ddq'):
        df[col] = df[col] * rad_s2_to_rpm2
    elif col.startswith('dq'):
        df[col] = df[col] * rad_s_to_rpm
    elif col.startswith('q'):
        df[col] = df[col] * rad_to_deg

# Save the converted data to a new file
df.to_csv('trajectory_pso_converted.csv', index=False)
print("Conversion complete. Saved as 'trajectory_pso_converted.csv'")