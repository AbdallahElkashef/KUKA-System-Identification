import numpy as np

NDOF  = 6
NF    = 5
WF    = 2.0*np.pi*0.1
FS    = 150.0

N_PER = int(round((2*np.pi/WF)*FS))
T_VEC = np.linspace(0, 2*np.pi/WF, N_PER, endpoint=False)

PPJ   = 2*NF+1
NDIM  = NDOF*PPJ
TORQUE_VARIANCES = np.array([1, 1, 1, 1, 1, 1]) 

Q_MIN  = np.array([-np.pi/2, -1.1, -1.1, -6.1086,-2.268,-6.1086])
Q_MAX  = np.array([ np.pi/2, 1.1 , 1.1, 6.1086, 2.268, 6.1086])
DQ_MAX = np.array([1.256,1.256,1.256,2.23,2.23,3.14])
DDQ_MAX= np.full(6, 15.0)
WS_Z_MIN =  0.050   # minimum height above base (table avoidance)
WS_Z_MAX =  0.900   # maximum height (UR5e reach ~0.85 m)
WS_R_MIN =  0.150   # minimum planar reach (avoid base cylinder)
WS_R_MAX =  0.850   # maximum planar reach

WS_Z_MIN =  0.050   # minimum height above base (table avoidance)
WS_Z_MAX =  2.326   # maximum height (UR5e reach ~0.85 m)
WS_R_MIN =  0.705   # minimum planar reach (avoid base cylinder)
WS_R_MAX =  1.911   # maximum planar reach



PEN_W     = 1e4
WS_PEN_W  = 1e4   
