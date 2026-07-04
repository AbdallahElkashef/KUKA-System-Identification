import numpy as np
from config import *
from trajectory import decode
from regressor_base import regressor_base
from f_k_constrains import forward_kinematics

def workspace_penalty(q_vec):
    _, _, z, r = forward_kinematics(q_vec)
    pen = 0.0
    pen += max(0.0, WS_Z_MIN - z) ** 2
    pen += max(0.0, z - WS_Z_MAX) ** 2
    pen += max(0.0, WS_R_MIN - r) ** 2
    pen += max(0.0, r - WS_R_MAX) ** 2
    return pen

def build_F(x):
    joints = decode(x)
    rows = []
    for t in T_VEC:
        rows.append(regressor_base(
            [j.q(t)   for j in joints],
            [j.dq(t)  for j in joints],
            [j.ddq(t) for j in joints]
        ))
    return np.vstack(rows)



def cond_F(F):
    n = np.linalg.norm(F, axis=0)
    n[n < 1e-12] = 1.0
    sv = np.linalg.svd(F / n, compute_uv=False)
    return sv[0] / max(sv[-1], 1e-12)

def penalty(x):
    joints = decode(x)
    pen = 0.0
    for t in T_VEC:
        q_t   = [jt.q(t)   for jt in joints]
        dq_t  = [jt.dq(t)  for jt in joints]
        ddq_t = [jt.ddq(t) for jt in joints]

        # ── 1. Joint-space limits ────────────────────────────────────────────
        for j in range(NDOF):
            pen += max(0, Q_MIN[j]  - q_t[j])  ** 2
            pen += max(0, q_t[j]   - Q_MAX[j]) ** 2
            pen += max(0, abs(dq_t[j])  - DQ_MAX[j])  ** 2
            pen += max(0, abs(ddq_t[j]) - DDQ_MAX[j]) ** 2

        # ── 2. Cartesian workspace limits ────────────────────────────────────
        pen += workspace_penalty(q_t)

    return pen

# def objective(x):
#     F = build_F(x)
#     return cond_F(F) + PEN_W * penalty(x)


SIGMA_WEIGHTS = 1.0 / np.sqrt(np.array(TORQUE_VARIANCES))

def objective(x):
    """
    Criterion 2: D-Optimality (Maximum-Likelihood)
    Minimizes the theoretical uncertainty of the physical parameters.
    """
    F = build_F(x)
    
    # 1. Apply the \Sigma^{-0.5} weights dynamically for NDOF
    N_time_steps = F.shape[0] // NDOF
    
    # Use np.tile if build_F rows are [t1_j1..6, t2_j1..6]
    # Use np.repeat if build_F rows are [t1..N_j1, t1..N_j2]
    W_vector = np.tile(SIGMA_WEIGHTS, N_time_steps) 
    
    F_weighted = F * W_vector[:, np.newaxis]
    
    # 2. Calculate the log determinant of the UNNORMALIZED weighted matrix
    sv_unnorm = np.linalg.svd(F_weighted, compute_uv=False)
    log_det = 2.0 * np.sum(np.log(np.maximum(sv_unnorm, 1e-300)))
    
    # 3. Minimize the negative log determinant + penalties
    return -log_det + PEN_W * penalty(x)



def make_bounds():
    lb, ub = [], []
    for j in range(NDOF):
        amp = float(DQ_MAX[j])
        lb += [-amp]*NF + [-amp]*NF + [float(Q_MIN[j])]
        ub += [ amp]*NF + [ amp]*NF + [float(Q_MAX[j])]
    return np.array(lb), np.array(ub)