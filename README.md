# KUKA KR 16 L6 Dynamic Parameter Identification

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![MuJoCo](https://img.shields.io/badge/MuJoCo-Simulation-red.svg)
![Pinocchio](https://img.shields.io/badge/Pinocchio-Dynamics-green.svg)
![SciPy](https://img.shields.io/badge/SciPy-Optimization-lightblue.svg)

## Overview
This repository contains the complete computational pipeline for the dynamic system identification of a 6-DoF KUKA KR 16 L6 industrial robotic arm. The project encompasses the design of highly constrained optimal excitation trajectories, high-fidelity physical simulation, advanced signal processing, and robust statistical parameter extraction. 

This work was developed as a Mechatronics and Robotics Engineering graduation project at Assiut University by Abdalla Amr Elkashef Abdelmaksoud Ibrahim and Mohamed Abdelmonem.

By mathematically decoupling rigid-body inertia from non-linear passive resistance, this pipeline successfully reduces 72 theoretical dynamic parameters to a minimal identifiable base set of 53, achieving highly accurate joint torque predictions with an average dynamic error of ~2.1% against real-world Rated Maximum Torques.

---

## Mathematical Foundation

### 1. Inverse Dynamics (Newton-Euler Formulation)
The rigid-body dynamics of the 6-DoF serial manipulator are established using the Newton-Euler formulation from the perspective of force equilibrium:

$$\tau = M(q)\ddot{q} + H(q, \dot{q}) + G(q) + \xi$$

Where:
* $q, \dot{q}, \ddot{q} \in \mathbb{R}^n$ represent joint position, velocity, and acceleration.
* $M(q) \in \mathbb{R}^{n \times n}$ is the configuration-dependent inertia matrix.
* $H(q, \dot{q}) \in \mathbb{R}^n$ contains the centrifugal and Coriolis forces.
* $G(q) \in \mathbb{R}^n$ represents the gravitational torques.
* $\xi \in \mathbb{R}^n$ encapsulates non-linear uncertainties such as joint friction.

### 2. Friction Modeling
To account for the continuous metal-on-metal scraping of bearings and speed-dependent internal lubrication resistance, a Coulomb-viscous friction model is decoupled into the system:

$$\xi = f_v \dot{q} + f_c sgn(\dot{q}) + \epsilon(q, \dot{q}, \ddot{q})$$

### 3. Base Parameter Reduction
A standard 6-DoF manipulator possesses $6 \times 12 = 72$ standard dynamic parameters. Due to structural unidentifiability (e.g., gravity blindness and rigid kinematic coupling), this pipeline utilizes a randomized Singular Value Decomposition (SVD) algorithm to mathematically group linearly dependent variables, reducing the system to a minimal set of **53 fully identifiable base parameters**.

---

## Pipeline Architecture

### Phase 1: Trajectory Optimization (`1_trajectory_optimization/`)
To properly excite the dynamic parameters, finite Fourier series trajectories are generated. The coefficients are optimized using a parallelized, stochastic Multi-Start **Sequential Least Squares Programming (SLSQP)** solver.
* **Objective:** Minimize the condition number of the global observation matrix ($cond(\Phi_M) = \sigma_{max} / \sigma_{min}$).
* **Constraints:** The solver utilizes the Jacobian to mathematically "surf" strict physical constraints, applying massive penalty multipliers ($50,000.0$) to avoid violating safe hemisphere boundaries or exceeding the $647^\circ/s$ maximum joint velocity.

### Phase 2: High-Fidelity Simulation (`2_simulation/`)
Because physical testing was constrained, the optimized trajectories are deployed into a pristine `mujoco` physics simulation.
* **Tracking:** An ultra-stiff Proportional-Derivative (PD) controller ensures flawless trajectory tracking (zero phase lag).
* **Data Trimming:** The initial $0.5s$ and final $0.5s$ of the 8-cycle simulation are algorithmically chopped to eliminate non-periodic acceleration transients from the dataset.

### Phase 3: Signal Processing & Parameter Extraction (`3_parameter_extraction/`)
The simulated data undergoes rigorous processing before mathematical extraction:
* **Kinematic Filtering:** A 4th-order zero-phase Butterworth low-pass filter ($\omega_n = 0.1$) is applied via `scipy.signal.filtfilt` to perfectly preserve mathematical amplitudes without introducing temporal phase shifts.
* **Torque Smoothing:** Motor torques are processed using a Savitzky-Golay filter (window 51, polynomial order 2) to mathematically hug the physical torque peaks while rejecting high-frequency controller noise.
* **Safe Normalized OLS:** The observation matrix is safely normalized to prevent scaling explosions of unobservable parameters, followed by an Ordinary Least Squares (OLS) extraction via Moore-Penrose pseudo-inverse.

---

## Results & Verification
The identified 53 base parameters were statistically validated against a completely unseen verification trajectory featuring a shifted fundamental frequency ($\omega_f = 2\pi(1/30)$). 

The physical model achieved highly precise torque predictions, drastically outperforming the standard literature benchmark of 10%:

| Joint # | Max Torque (MT) | RMSE of Physical Model | Percentage of MT (%) |
| :--- | :--- | :--- | :--- |
| **Joint 1** | 300 Nm | 1.966 | 0.66% |
| **Joint 2** | 300 Nm | 2.462 | 0.82% |
| **Joint 3** | 150 Nm | 1.002 | 0.67% |
| **Joint 4** | 50 Nm | 2.948 | 5.90% |
| **Joint 5** | 50 Nm | 1.475 | 2.95% |
| **Joint 6** | 10 Nm | 0.151 | 1.51% |

**Overall Performance:** The global execution yielded an L2-Norm Error of 10.92%, translating to an average dynamic physical error of exactly **2.08%**.

---

## Repository Structure

```text
kuka_dynamic_identification/
│
├── 1_trajectory_optimization/
│   ├── optimize_condition.py       # Parallelized multi-start SLSQP solver
│   ├── generate_verification.py    # Generates unseen validation trajectories
│   └── evaluate_trajectory.py      # Calculates exact penalty breakdowns
│
├── 2_simulation/
│   ├── simulate_environment.py     # MuJoCo identification data gathering
│   └── simulate_verification.py    # MuJoCo unseen validation data gathering
│
├── 3_parameter_extraction/
│   ├── least_squares_solver.py     # Safe Normalized OLS & filtering
│   ├── verify_parameters.py        # RMSE and %MT mathematical verification
│   ├── generate_confidence_table.py# Covariance and SVD statistical mapping
│   ├── plot_kinematic_phase_lag.py # Visualizes PD controller tracking
│   └── plot_verification_torques.py# Academic plotting for predicted vs measured
│
├── data/                           # Output directory for trajectories, NPZ datasets, and plots
├── robot_assets/                   # KUKA KR16 L6 URDF, XML, and STL meshes
└── utils/                          
    ├── base_params_calculator.py   # Randomized SVD rank revelation logic
    ├── regressor_builder.py        # Pinocchio spatial inertia/friction fusion
    └── trajectory_math.py          # Fourier series generation math

## Execution Pipeline

To replicate the findings or apply this pipeline to your own robotic model, execute the modules sequentially. The `utils` directory is automatically routed in the system path for all scripts, so you do not need to manually configure environment variables.

### Step 1: Trajectory Optimization

Generate the highly constrained Fourier series coefficients for both the identification and verification phases.

```bash
cd 1_trajectory_optimization
python optimize_condition.py
python generate_verification.py
```

> **Note:** The optimizer utilizes a `concurrent.futures` multi-processing engine. It will automatically scale to utilize your available CPU cores to execute the stochastic Multi-Start SLSQP trials.

### Step 2: High-Fidelity Simulation

Deploy the optimized mathematical coefficients into the MuJoCo physics engine to extract clean, zero-phase-lag datasets.

```bash
cd ../2_simulation
python simulate_environment.py
python simulate_verification.py
```

This will launch the MuJoCo passive viewer. The simulated KUKA manipulator will execute 8 cycles of the trajectories, automatically trim the initial and final transient phases, and save the steady-state data (`.npz`) to the `data/training_datasets/` directory.

### Step 3: Parameter Extraction & Validation

Filter the collected datasets, build the safely normalized observation matrix, and extract the 53 minimum base parameters.

```bash
cd ../3_parameter_extraction
python least_squares_solver.py
```

To mathematically validate the extracted parameters and generate the academic plots:

```bash
python generate_confidence_table.py
python verify_parameters.py
python plot_verification_torques.py
```

High-resolution verification plots and parameter arrays will be saved to the `data/` directory.

---

## Future Work

* **Physically Consistent Parameter Identification (PCPI):** While the current Ordinary Least Squares (OLS) solver provides highly accurate torque predictions, the integration of Semi-Definite Programming (SDP) or Linear Matrix Inequalities (LMI) is planned to mathematically bound the extracted masses and inertias to strict, physically realistic positive-definite values.
* **Hardware Deployment:** Transitioning the verified mathematical model from the MuJoCo simulation environment directly to the physical KUKA KR 16 L6 hardware testbed.

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

clean this and check if any additional details should be added. only give me a single markdown code. no additional text beyond the code
