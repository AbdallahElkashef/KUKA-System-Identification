import numpy as np
import pinocchio as pin

from regressor_builder import build_observation_matrix

def get_base_parameter_count(model, data):
    """
    Dynamically calculates the number of identifiable base parameters
    for a given robot model using SVD on a randomized Regressor matrix,
    including Coulomb and Viscous friction.
    """
    print("Dynamically calculating base parameter count (with friction)...")
    
    num_samples = 100 
    Phi_M = []
    
    # 1. Generate a random "dummy" trajectory
    for _ in range(num_samples):
        q = pin.randomConfiguration(model)
        dq = np.random.uniform(-2.0, 2.0, model.nv) 
        ddq = np.random.uniform(-5.0, 5.0, model.nv)
        
        # Calculate the 6x60 rigid-body regressor
        phi_inertial = pin.computeJointTorqueRegressor(model, data, q, dq, ddq)
        
        # Build the 6x12 friction regressor
        phi_fric = np.zeros((6, 12))
        for j in range(6):
            phi_fric[j, 2*j] = dq[j]
            phi_fric[j, 2*j + 1] = np.sign(dq[j])
            
        # Fuse them into 6x72
        phi_total = np.hstack((phi_inertial, phi_fric))
        Phi_M.append(phi_total)
        
    Phi_M = np.vstack(Phi_M)
    
    # 2. Normalize columns 
    col_norms = np.linalg.norm(Phi_M, axis=0)
    col_norms[col_norms == 0] = 1.0 
    Phi_norm = Phi_M / col_norms
    
    # 3. Perform SVD to find the mathematical rank
    s = np.linalg.svd(Phi_norm, compute_uv=False)
    
    # 4. Count observable parameters
    tolerance = 1e-7
    base_param_count = np.sum(s > tolerance)
    
    print(f"-> Discovered {base_param_count} identifiable base parameters out of {model.nv * 12} total parameters.")
    
    return base_param_count