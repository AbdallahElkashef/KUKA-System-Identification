import numpy as np
import pinocchio as pin

def build_observation_matrix(model, data, q, dq, ddq):
    """
    Constructs the observation matrix (Regressor) W for a trajectory,
    including rigid-body inertia, Coulomb friction, and Viscous friction.
    """
    W = []
    
    # Ensure inputs are 2D arrays even if a single state is passed
    q = np.atleast_2d(q)
    dq = np.atleast_2d(dq)
    ddq = np.atleast_2d(ddq)
    
    for i in range(len(q)):
        phi_inertial = pin.computeJointTorqueRegressor(model, data, q[i], dq[i], ddq[i])
        phi_fric = np.zeros((6, 12))
        for j in range(6):
            phi_fric[j, 2*j] = dq[i, j]               
            phi_fric[j, 2*j + 1] = np.sign(dq[i, j])  
            
        phi_total = np.hstack((phi_inertial, phi_fric))
        W.append(phi_total)
        
    return np.vstack(W)