import numpy as np
import os
import matplotlib.pyplot as plt

from config import *
from trajectory import decode
from problem import build_F, penalty


def _compute_metrics(F):
    """
    MLE / D-optimality metrics

    Returns
    -------
    sv_w      : singular values of weighted regressor
    log_det   : log det(Fᵀ Σ⁻¹ F)
    rank_w    : rank(F_weighted)
    """

    sigma_weights = 1.0 / np.sqrt(np.array(TORQUE_VARIANCES))

    n_time_steps = F.shape[0] // NDOF

    W_vector = np.tile(sigma_weights, n_time_steps)

    F_weighted = F * W_vector[:, np.newaxis]

    sv_w = np.linalg.svd(F_weighted, compute_uv=False)

    log_det = 2.0 * np.sum(
        np.log(np.maximum(sv_w, 1e-300))
    )

    rank_w = np.linalg.matrix_rank(F_weighted)

    return sv_w, log_det, rank_w


def report_and_plot(best_f, best_x, history, label):

    joints = decode(best_x)

    F = build_F(best_x)

    pen = penalty(best_x)

    sv, log_det, rank_w = _compute_metrics(F)

    # ==========================================================
    # Console Report
    # ==========================================================
    print("\n" + "=" * 70)
    print(f"RESULT [{label}]")
    print("=" * 70)

    print(
        f"Objective value      : "
        f"{best_f - PEN_W * pen:.6f}"
    )

    print(
        f"log det(FᵀΣ⁻¹F)      : "
        f"{log_det:.6f}"
    )

    print(
        f"det(FIM)             : "
        f"{np.exp(min(log_det,700)):.6e}"
    )

    print(
        f"sigma_max            : "
        f"{sv[0]:.6e}"
    )

    print(
        f"sigma_min            : "
        f"{sv[-1]:.6e}"
    )

    print(
        f"rank(F_weighted)     : "
        f"{rank_w}/{F.shape[1]}"
    )

    print(
        f"F shape              : "
        f"{F.shape}"
    )

    if pen < 1e-4:
        print("Constraints          : FEASIBLE OK")
    else:
        print(f"Constraints          : INFEASIBLE (pen={pen:.6g})")

    print("\nJoint Parameters")
    print("-" * 70)

    for j, jt in enumerate(joints):

        print(
            f"J{j+1}: "
            f"q0={np.degrees(jt.q0):8.3f} deg, "
            f"max|a|={np.max(np.abs(jt.a)):10.5f}, "
            f"max|b|={np.max(np.abs(jt.b)):10.5f}"
        )

    print("=" * 70)

    # ==========================================================
    # Plotting
    # ==========================================================
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    fig.suptitle(
        f"MLE / D-Optimal Excitation [{label}]\n"
        f"log det(FᵀΣ⁻¹F) = {log_det:.2f}    "
        f"rank = {rank_w}/{F.shape[1]}",
        fontsize=13
    )

    # ----------------------------------------------------------
    # Convergence
    # ----------------------------------------------------------
    ax = axes[0]

    ax.plot(history, lw=2)

    ax.set_xlabel("Iteration")

    ax.set_ylabel("Objective")

    ax.set_title("Convergence")

    ax.grid(True)

    # ----------------------------------------------------------
    # Trajectory Plots
    # ----------------------------------------------------------
    t_plot = np.linspace(0, 2 * np.pi / WF, 1500)

    colors = plt.cm.tab10(np.linspace(0, 1, NDOF))

    # Position
    ax = axes[1]

    for j, jt in enumerate(joints):

        ax.plot(
            t_plot,
            np.degrees([jt.q(t) for t in t_plot]),
            color=colors[j],
            label=f"J{j+1}"
        )

    ax.set_title("Joint Position")

    ax.set_xlabel("Time (s)")

    ax.set_ylabel("deg")

    ax.grid(True)

    ax.legend()

    # Velocity
    ax = axes[2]

    for j, jt in enumerate(joints):

        ax.plot(
            t_plot,
            np.degrees([jt.dq(t) for t in t_plot]),
            color=colors[j],
            label=f"J{j+1}"
        )

    ax.set_title("Joint Velocity")

    ax.set_xlabel("Time (s)")

    ax.set_ylabel("deg/s")

    ax.grid(True)

    ax.legend()

    # Acceleration
    ax = axes[3]

    for j, jt in enumerate(joints):

        ax.plot(
            t_plot,
            np.degrees([jt.ddq(t) for t in t_plot]),
            color=colors[j],
            label=f"J{j+1}"
        )

    ax.set_title("Joint Acceleration")

    ax.set_xlabel("Time (s)")

    ax.set_ylabel("deg/s²")

    ax.grid(True)

    ax.legend()

    plt.tight_layout()

    # ==========================================================
    # Save outputs
    # ==========================================================
    out_dir = os.path.join(os.getcwd(), "outputs")

    os.makedirs(out_dir, exist_ok=True)

    tag = label.replace(" ", "_").lower()

    # PNG
    out_png = os.path.join(
        out_dir,
        f"trajectory_{tag}.png"
    )

    plt.savefig(
        out_png,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ==========================================================
    # CSV trajectory
    # ==========================================================
    t_csv = np.linspace(
        0,
        2*np.pi/WF,
        1500
    )

    rows = []

    for t in t_csv:

        row = [t]

        row += [jt.q(t) for jt in joints]

        row += [jt.dq(t) for jt in joints]

        row += [jt.ddq(t) for jt in joints]

        rows.append(row)

    rows = np.array(rows)

    headers = (
        ["time"]
        + [f"q{i+1}" for i in range(NDOF)]
        + [f"dq{i+1}" for i in range(NDOF)]
        + [f"ddq{i+1}" for i in range(NDOF)]
    )

    out_csv = os.path.join(
        out_dir,
        f"trajectory_{tag}.csv"
    )

    np.savetxt(
        out_csv,
        rows,
        delimiter=",",
        header=",".join(headers),
        comments=""
    )

    # ==========================================================
    # Save optimization results
    # ==========================================================
    data = {

        "wf": np.array([WF]),

        "nf": np.array([NF]),

        "log_det_FIM": np.array([log_det]),

        "rank_F_weighted": np.array([rank_w]),

        "sigma_max": np.array([sv[0]]),

        "sigma_min": np.array([sv[-1]]),

        "objective": np.array([best_f]),

        "penalty": np.array([pen]),
    }

    for j, jt in enumerate(joints):

        data[f"a_{j}"] = jt.a

        data[f"b_{j}"] = jt.b

        data[f"q0_{j}"] = np.array([jt.q0])

    out_npz = os.path.join(
        out_dir,
        f"trajectory_{tag}.npz"
    )

    np.savez(out_npz, **data)

    print(f"\nPlot   -> {out_png}")
    print(f"CSV    -> {out_csv}")
    print(f"Params -> {out_npz}")