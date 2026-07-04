import numpy as np
import pinocchio as pin

def get_base_parameter_count(model, data):
    """
    Dynamically calculates the number of identifiable base parameters
    for a given robot model using SVD on a randomized Regressor matrix.
    """
    print("Dynamically calculating base parameter count...")
    
    num_samples = 100 
    Phi_M = []
    
    # 1. Generate a random "dummy" trajectory
    for _ in range(num_samples):
        # Generate random joint positions, velocities, and accelerations
        q = pin.randomConfiguration(model)
        dq = np.random.uniform(-2.0, 2.0, model.nv) 
        ddq = np.random.uniform(-5.0, 5.0, model.nv)
        
        # Calculate the 6x60 regressor for this random instant
        phi = pin.computeJointTorqueRegressor(model, data, q, dq, ddq)
        Phi_M.append(phi)
        
    Phi_M = np.vstack(Phi_M)
    
    # 2. Normalize columns (same as your objective function)
    col_norms = np.linalg.norm(Phi_M, axis=0)
    col_norms[col_norms == 0] = 1.0 
    Phi_norm = Phi_M / col_norms
    
    # 3. Perform SVD to find the mathematical rank
    s = np.linalg.svd(Phi_norm, compute_uv=False)
    
    # 4. Count how many singular values are above numerical noise
    # Any stretch larger than 1e-7 is considered a real, observable parameter
    tolerance = 1e-7
    base_param_count = np.sum(s > tolerance)
    
    print(f"-> Discovered {base_param_count} identifiable base parameters out of {model.nv * 10} total parameters.")
    
    return base_param_count