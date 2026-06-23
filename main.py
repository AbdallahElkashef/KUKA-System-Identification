import time
from config import *
from problem import objective, make_bounds
from pso import PSO
from report import report_and_plot

def main():
    PARTICLES = 150
    ITERS     = 1500
    SEED      = 42

    lb, ub = make_bounds()

    print("="*60)
    print("Trajectory Optimization (PSO)")
    print("="*60)

    t0 = time.time()

    opt = PSO(n=PARTICLES, T=ITERS, lb=lb, ub=ub, seed=SEED)
    best_f, best_x = opt.optimise(objective)

    print(f"\nWall time: {time.time()-t0:.2f}s")

    report_and_plot(best_f, best_x, opt.history, "PSO")

if __name__ == "__main__":
    main()