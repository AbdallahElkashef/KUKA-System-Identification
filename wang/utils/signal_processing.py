import numpy as np
from scipy.signal import butter, filtfilt

def filter_and_trim_data(t, q, dq, tau, cutoff_freq=0.1, filter_order=4, trim_sec=0.5):
    """
    Calculates acceleration, applies a zero-phase Butterworth filter to 
    velocity, acceleration, and measured torque, and trims edge transients.
    """
    dt = t[1] - t[0]
    ddq_raw = np.gradient(dq, dt, axis=0)
    
    # Create the Butterworth low-pass filter
    b, a = butter(filter_order, cutoff_freq, btype='low') 
    
    # Apply filter to kinematics and measured torques (removes PWM noise)
    ddq_filt = filtfilt(b, a, ddq_raw, axis=0)
    dq_filt = filtfilt(b, a, dq, axis=0) 
    tau_filt = filtfilt(b, a, tau, axis=0) 

    # Edge-effect trim
    trim = int(trim_sec / dt)
    if trim > 0:
        t = t[trim:-trim]
        q = q[trim:-trim]
        dq_filt = dq_filt[trim:-trim]
        ddq_filt = ddq_filt[trim:-trim]
        tau_filt = tau_filt[trim:-trim]
        
    return t, q, dq_filt, ddq_filt, tau_filt