import numpy as np
from config import *

class FourierJoint:
    def __init__(self, wf, a, b, q0, nf):
        self.wf = float(wf)
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.q0 = float(q0)
        self.nf = int(nf)

    def q(self, t):
        v = self.q0
        for l in range(1, self.nf+1):
            v += (self.a[l-1]/(self.wf*l))*np.sin(self.wf*l*t) \
               - (self.b[l-1]/(self.wf*l))*np.cos(self.wf*l*t)
        return v

    def dq(self, t):
        v = 0.0
        for l in range(1, self.nf+1):
            v += self.a[l-1]*np.cos(self.wf*l*t) + self.b[l-1]*np.sin(self.wf*l*t)
        return v

    def ddq(self, t):
        v = 0.0
        for l in range(1, self.nf+1):
            v += -self.a[l-1]*self.wf*l*np.sin(self.wf*l*t) \
               + self.b[l-1]*self.wf*l*np.cos(self.wf*l*t)
        return v


def decode(x):
    return [
        FourierJoint(
            WF,
            x[j*PPJ:j*PPJ+NF],
            x[j*PPJ+NF:j*PPJ+2*NF],
            x[j*PPJ+2*NF],
            NF
        )
        for j in range(NDOF)
    ]