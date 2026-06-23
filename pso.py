import numpy as np
import time
from config import *
class PSO:
    def __init__(self, n, T, lb, ub,
                 w_max=0.9, w_min=0.4, c1=1.494, c2=1.494,
                 seed=42, verbose=True, log_every=10):
        self.n=n; self.T=T
        self.lb=np.asarray(lb); self.ub=np.asarray(ub)
        self.w_max=w_max; self.w_min=w_min
        self.c1=c1; self.c2=c2
        self.verbose=verbose; self.log_every=log_every
        self.rng=np.random.default_rng(seed)
        self.history=[]

    def optimise(self, f):
        lb,ub=self.lb,self.ub; D=len(lb)
        vmax=0.2*(ub-lb)   # velocity_clamping
        X=self.rng.uniform(lb,ub,size=(self.n,D))
        V=self.rng.uniform(-vmax,vmax,size=(self.n,D))
        cost=np.array([f(X[i]) for i in range(self.n)])
        pbest_X=X.copy(); pbest_f=cost.copy()
        gi=np.argmin(pbest_f)
        gbest_X=pbest_X[gi].copy(); gbest_f=pbest_f[gi]
        self.history=[gbest_f]
        t0=time.time()
        for it in range(1,self.T+1):
            w=self.w_max-(self.w_max-self.w_min)*it/self.T
            r1=self.rng.random((self.n,D)); r2=self.rng.random((self.n,D))
            V=w*V+self.c1*r1*(pbest_X-X)+self.c2*r2*(gbest_X-X)
            V=np.clip(V,-vmax,vmax)
            X=np.clip(X+V,lb,ub)
            cost=np.array([f(X[i]) for i in range(self.n)])
            imp=cost<pbest_f; pbest_X[imp]=X[imp]; pbest_f[imp]=cost[imp]
            gi=np.argmin(pbest_f)
            if pbest_f[gi]<gbest_f:
                gbest_f=pbest_f[gi]; gbest_X=pbest_X[gi].copy()
            self.history.append(gbest_f)
            if self.verbose and (it%self.log_every==0 or it==1):
                print(f"  Iter {it:4d}/{self.T}  |  best cond = {gbest_f:.4f}"
                      f"  |  {time.time()-t0:.1f}s")
        return gbest_f,gbest_X
