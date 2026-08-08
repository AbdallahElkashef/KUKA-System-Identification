# KUKA KR 16 L6 Dynamic Parameter Identification

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![MuJoCo](https://img.shields.io/badge/MuJoCo-Simulation-red.svg)
![Pinocchio](https://img.shields.io/badge/Pinocchio-Dynamics-green.svg)
![SciPy](https://img.shields.io/badge/SciPy-Optimization-lightblue.svg)

## Overview
This repository contains the complete computational pipeline for the dynamic system identification of a 6-DoF KUKA KR 16 L6 industrial robotic arm. The project encompasses the design of highly constrained optimal excitation trajectories, high-fidelity physical simulation, advanced signal processing, and robust statistical parameter extraction. 

This work was developed as a Mechatronics and Robotics Engineering graduation project at Assiut University by Abdalla Amr Elkashef Abdelmaksoud Ibrahim and Mohamed Abdelmonem.

By mathematically decoupling rigid-body inertia from non-linear passive resistance, this pipeline successfully reduces 72 theoretical dynamic parameters to a minimal identifiable base set of 53, achieving highly accurate joint torque predictions with an average dynamic error of ~2.1% against real-world Rated Maximum Torques[cite: 12].

---

## Mathematical Foundation

### 1. Inverse Dynamics (Newton-Euler Formulation)
The rigid-body dynamics of the 6-DoF serial manipulator are established using the Newton-Euler formulation from the perspective of force equilibrium[cite: 12]:

$$\tau = M(q)\ddot{q} + H(q, \dot{q}) + G(q) + \xi$$

Where:
* $q, \dot{q}, \ddot{q} \in \mathbb{R}^n$ represent joint position, velocity, and acceleration[cite: 12].
* $M(q) \in \mathbb{R}^{n \times n}$ is the configuration-dependent inertia matrix[cite: 12].
* $H(q, \dot{q}) \in \mathbb{R}^n$ contains the centrifugal and Coriolis forces[cite: 12].
* $G(q) \in \mathbb{R}^n$ represents the gravitational torques[cite: 12].
* $\xi \in \mathbb{R}^n$ encapsulates non-linear uncertainties such as joint friction[cite: 12].

### 2. Friction Modeling
To account for the continuous metal-on-metal scraping of bearings and speed-dependent internal lubrication resistance, a Coulomb-viscous friction model is decoupled into the system[cite: 12]:

$$\xi = f_v \dot{q} + f_c sgn(\dot{q}) + \epsilon(q, \dot{q}, \ddot{q})$$

### 3. Base Parameter Reduction
A standard 6-DoF manipulator possesses $6 \times 12 = 72$ standard dynamic parameters[cite: 12]. Due to structural unidentifiability (e.g., gravity blindness and rigid kinematic coupling), this pipeline utilizes a randomized Singular Value Decomposition (SVD) algorithm to mathematically group linearly dependent variables, reducing the system to a minimal set of **53 fully identifiable base parameters**[cite: 12].

---

## Pipeline Architecture

### Phase 1: Trajectory Optimization (`1_trajectory_optimization/`)
To properly excite the dynamic parameters, finite Fourier series trajectories are generated[cite: 12]. The coefficients are optimized using a parallelized, stochastic Multi-Start **Sequential Least Squares Programming (SLSQP)** solver[cite: 3].
* **Objective:** Minimize the condition number of the global observation matrix ($cond(\Phi_M) = \sigma_{max} / \sigma_{min}$)[cite: 12].
* **Constraints:** The solver utilizes the Jacobian to mathematically "surf" strict physical constraints, applying massive penalty multipliers ($50,000.0$) to avoid violating safe hemisphere boundaries or exceeding the $647^\circ/s$ maximum joint velocity[cite: 3, 12].

### Phase 2: High-Fidelity Simulation (`2_simulation/`)
Because physical testing was constrained, the optimized trajectories are deployed into a pristine `mujoco` physics simulation[cite: 5, 12].
* **Tracking:** An ultra-stiff Proportional-Derivative (PD) controller ensures flawless trajectory tracking (zero phase lag)[cite: 5, 12].
* **Data Trimming:** The initial $0.5s$ and final $0.5s$ of the 8-cycle simulation are algorithmically chopped to eliminate non-periodic acceleration transients from the dataset[cite: 7, 12].

### Phase 3: Signal Processing & Parameter Extraction (`3_parameter_extraction/`)
The simulated data undergoes rigorous processing before mathematical extraction:
* **Kinematic Filtering:** A 4th-order zero-phase Butterworth low-pass filter ($\omega_n = 0.1$) is applied via `scipy.signal.filtfilt` to perfectly preserve mathematical amplitudes without introducing temporal phase shifts[cite: 7, 12].
* **Torque Smoothing:** Motor torques are processed using a Savitzky-Golay filter (window 51, polynomial order 2) to mathematically hug the physical torque peaks while rejecting high-frequency controller noise[cite: 7, 12].
* **Safe Normalized OLS:** The observation matrix is safely normalized to prevent scaling explosions of unobservable parameters, followed by an Ordinary Least Squares (OLS) extraction via Moore-Penrose pseudo-inverse[cite: 7, 12].

---

## Results & Verification
The identified 53 base parameters were statistically validated against a completely unseen verification trajectory featuring a shifted fundamental frequency ($\omega_f = 2\pi(1/30)$)[cite: 6, 12]. 

The physical model achieved highly precise torque predictions, drastically outperforming the standard literature benchmark of 10%[cite: 12]:

| Joint # | Max Torque (MT) | RMSE of Physical Model | Percentage of MT (%) |
| :--- | :--- | :--- | :--- |
| **Joint 1** | 300 Nm | 1.966 | 0.66% |
| **Joint 2** | 300 Nm | 2.462 | 0.82% |
| **Joint 3** | 150 Nm | 1.002 | 0.67% |
| **Joint 4** | 50 Nm | 2.948 | 5.90% |
| **Joint 5** | 50 Nm | 1.475 | 2.95% |
| **Joint 6** | 10 Nm | 0.151 | 1.51% |

**Overall Performance:** The global execution yielded an L2-Norm Error of 10.92%, translating to an average dynamic physical error of exactly **2.08%**[cite: 12].

---

## Repository Structure

```text
kuka_dynamic_identification/
│
├── 1_trajectory_optimization/
│   ├── optimize_condition.py       # Parallelized multi-start SLSQP solver
│   ├── generate_verification.py    # Generates unseen validation trajectories
│   └── evaluate_trajectory.py      # Calculates exact penalty breakdowns
│
├── 2_simulation/
│   ├── simulate_environment.py     # MuJoCo identification data gathering
│   └── simulate_verification.py    # MuJoCo unseen validation data gathering
│
├── 3_parameter_extraction/
│   ├── least_squares_solver.py     # Safe Normalized OLS & filtering
│   ├── verify_parameters.py        # RMSE and %MT mathematical verification
│   ├── generate_confidence_table.py# Covariance and SVD statistical mapping
│   ├── plot_kinematic_phase_lag.py # Visualizes PD controller tracking
│   └── plot_verification_torques.py# Academic plotting for predicted vs measured
│
├── data/                           # Output directory for trajectories, NPZ datasets, and plots
├── robot_assets/                   # KUKA KR16 L6 URDF, XML, and STL meshes
└── utils/                          
    ├── base_params_calculator.py   # Randomized SVD rank revelation logic
    ├── regressor_builder.py        # Pinocchio spatial inertia/friction fusion
    └── trajectory_math.py          # Fourier series generation math
