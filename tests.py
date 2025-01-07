import numpy as np
import os
from numpy.random import default_rng
from scipy.stats import uniform, norm

from hista import *

rng = default_rng(10)  # Seed the random number generator

init_t_with_L = False

# Helper functions for sampling
def sample_no_replace(arr, size):
    return rng.choice(arr, size=size, replace=False)


# Parameters
sigma = 1
gamma = 1
lam = 0.01

# Generate data
A = rng.normal(0, 1, (100, 10))
x_true = np.zeros(10)
nonzero_indices = sample_no_replace(np.arange(10), 3)
x_true[nonzero_indices] = uniform.rvs(-10, 20, size=3, random_state=rng)
epsilon = norm.rvs(0, sigma**2, size=(100, 1), random_state=rng)

b = A @ x_true + epsilon.flatten()
outlier_indices = sample_no_replace(np.arange(100), 5)
b[outlier_indices] = uniform.rvs(50, 50, size=5, random_state=rng)

# Run algorithms
x_hista, x_hista_vals = HISTA(A, b, lam, gamma)
x_hista_ls, x_hista_ls_vals = HISTA(A, b, lam, gamma, line_search=True)
x_fasthista, x_fasthista_vals = FastHISTA(A, b, lam, gamma, line_search=False, init_t_with_L=init_t_with_L)
x_fasthista_ls, x_fasthista_ls_vals = FastHISTA(A, b, lam, gamma, line_search=True, init_t_with_L=init_t_with_L)
x_proxnewton_ls, x_proxnewton_ls_vals = ProxNewton(A, b, lam, gamma, line_search=True)
x_fista, x_fista_vals = FISTA(A, b, lam)
x_fista_ls, x_fista_ls_vals = FISTA(A, b, lam, line_search=True)

# Organize losses
experiment_1_losses = {
    "HISTA": x_hista_vals,
    "HISTA_LS": x_hista_ls_vals,
    "FastHISTA": x_fasthista_vals,
    "FastHISTA_LS": x_fasthista_ls_vals,
}

experiment_1_losses_2 = {
    "ProxNewton_LS": x_proxnewton_ls_vals,
}

# Display results
print("Test 1")
print("True:", x_true)
print("True function value:", robust_huber(x_true, A, b, gamma, lam))
print("\nPred HISTA:", x_hista)
print("HISTA function value:", robust_huber(x_hista, A, b, gamma, lam))
print("\nPred HISTA w/ Line Search:", x_hista_ls)
print("HISTA function value w/ Line Search:", robust_huber(x_hista_ls, A, b, gamma, lam))
print("\nPred FastHISTA:", x_fasthista)
print("FastHISTA function value:", robust_huber(x_fasthista, A, b, gamma, lam))
print("\nPred FastHISTA w/ Line Search:", x_fasthista_ls)
print("FastHISTA function value w/ Line Search:", robust_huber(x_fasthista_ls, A, b, gamma, lam))
print("\nPred ProxNewton w/ Line Search:", x_proxnewton_ls)
print("ProxNewton function value w/ Line Search:", robust_huber(x_proxnewton_ls, A, b, gamma, lam))
print("\nPred FISTA:", x_fista)
print("FISTA function value:", robust_huber(x_fista, A, b, gamma, lam))
print("\nPred FISTA w/ Line Search:", x_fista_ls)
print("FISTA function value w/ Line Search:", robust_huber(x_fista_ls, A, b, gamma, lam))

# Parameters
A = rng.normal(0, 1, (200, 20))

x_true = np.zeros(20)
nonzero_indices = sample_no_replace(np.arange(20), 3)
x_true[nonzero_indices] = uniform.rvs(-10, 20, size=3, random_state=rng)

epsilon = norm.rvs(0, 0.1**2, size=(200, 1), random_state=rng)
outlier_indices = sample_no_replace(np.arange(100), 10)
epsilon[outlier_indices] = norm.rvs(0, 5**2, size=10, random_state=rng).reshape(-1, 1)

b = A @ x_true + epsilon.flatten()

# Run algorithms
x_hista, x_hista_vals = HISTA(A, b, lam, gamma)
x_hista_ls, x_hista_ls_vals = HISTA(A, b, lam, gamma, line_search=True)
x_fasthista, x_fasthista_vals = FastHISTA(A, b, lam, gamma, line_search=False, init_t_with_L=init_t_with_L)
x_fasthista_ls, x_fasthista_ls_vals = FastHISTA(A, b, lam, gamma, line_search=True, init_t_with_L=init_t_with_L)
x_proxnewton_ls, x_proxnewton_ls_vals = ProxNewton(A, b, lam, gamma, line_search=True)
x_fista, x_fista_vals = FISTA(A, b, lam)
x_fista_ls, x_fista_ls_vals = FISTA(A, b, lam, line_search=True)

# Organize losses
experiment_2_losses = {
    "HISTA": x_hista_vals,
    "HISTA_LS": x_hista_ls_vals,
    "FastHISTA": x_fasthista_vals,
    "FastHISTA_LS": x_fasthista_ls_vals,
}

experiment_2_losses_2 = {
    "ProxNewton_LS": x_proxnewton_ls_vals,
}

# Display results
print("Test 2")
print("True:", x_true)
print("True function value:", robust_huber(x_true, A, b, gamma, lam))
print("\nPred HISTA:", x_hista)
print("HISTA function value:", robust_huber(x_hista, A, b, gamma, lam))
print("\nPred HISTA w/ Line Search:", x_hista_ls)
print("HISTA function value w/ Line Search:", robust_huber(x_hista_ls, A, b, gamma, lam))
print("\nPred FastHISTA:", x_fasthista)
print("FastHISTA function value:", robust_huber(x_fasthista, A, b, gamma, lam))
print("\nPred FastHISTA w/ Line Search:", x_fasthista_ls)
print("FastHISTA function value w/ Line Search:", robust_huber(x_fasthista_ls, A, b, gamma, lam))
print("\nPred ProxNewton w/ Line Search:", x_proxnewton_ls)
print("ProxNewton function value w/ Line Search:", robust_huber(x_proxnewton_ls, A, b, gamma, lam))
print("\nPred FISTA:", x_fista)
print("FISTA function value:", robust_huber(x_fista, A, b, gamma, lam))
print("\nPred FISTA w/ Line Search:", x_fista_ls)
print("FISTA function value w/ Line Search:", robust_huber(x_fista_ls, A, b, gamma, lam))

# Parameters
A = np.zeros((100, 10))

# Populate sparse columns with 10% nonzero rows
for j in range(10):
    nonzero_indices = sample_no_replace(np.arange(100), 10)
    A[nonzero_indices, j] = rng.normal(0, 1, size=10)

# Add outliers to specific columns
outlier_columns = sample_no_replace(np.arange(10), 2)
for col in outlier_columns:
    outlier_indices = sample_no_replace(np.arange(100), 5)
    A[outlier_indices, col] = uniform.rvs(50, 50, size=5, random_state=rng)

x_true = np.zeros(10)
nonzero_indices = sample_no_replace(np.arange(10), 3)
x_true[nonzero_indices] = uniform.rvs(-10, 20, size=3, random_state=rng)

epsilon = norm.rvs(0, sigma**2, size=100, random_state=rng)
b = A @ x_true + epsilon

# Run algorithms
x_hista, x_hista_vals = HISTA(A, b, lam, gamma)
x_hista_ls, x_hista_ls_vals = HISTA(A, b, lam, gamma, line_search=True)
x_fasthista, x_fasthista_vals = FastHISTA(A, b, lam, gamma, line_search=False, init_t_with_L=init_t_with_L)
x_fasthista_ls, x_fasthista_ls_vals = FastHISTA(A, b, lam, gamma, line_search=True, init_t_with_L=init_t_with_L)
x_proxnewton_ls, x_proxnewton_ls_vals = ProxNewton(A, b, lam, gamma, line_search=True)
x_fista, x_fista_vals = FISTA(A, b, lam)
x_fista_ls, x_fista_ls_vals = FISTA(A, b, lam, line_search=True)

# Organize losses
experiment_3_losses = {
    "HISTA": x_hista_vals,
    "HISTA_LS": x_hista_ls_vals,
    "FastHISTA": x_fasthista_vals,
    "FastHISTA_LS": x_fasthista_ls_vals,
}

experiment_3_losses_2 = {
    "ProxNewton_LS": x_proxnewton_ls_vals,
}

# Display results
print("Test 3")
print("True:", x_true)
print("True function value:", robust_huber(x_true, A, b, gamma, lam))
print("\nPred HISTA:", x_hista)
print("HISTA function value:", robust_huber(x_hista, A, b, gamma, lam))
print("\nPred HISTA w/ Line Search:", x_hista_ls)
print("HISTA function value w/ Line Search:", robust_huber(x_hista_ls, A, b, gamma, lam))
print("\nPred FastHISTA:", x_fasthista)
print("FastHISTA function value:", robust_huber(x_fasthista, A, b, gamma, lam))
print("\nPred FastHISTA w/ Line Search:", x_fasthista_ls)
print("FastHISTA function value w/ Line Search:", robust_huber(x_fasthista_ls, A, b, gamma, lam))
print("\nPred ProxNewton w/ Line Search:", x_proxnewton_ls)
print("ProxNewton function value w/ Line Search:", robust_huber(x_proxnewton_ls, A, b, gamma, lam))
print("\nPred FISTA:", x_fista)
print("FISTA function value:", robust_huber(x_fista, A, b, gamma, lam))
print("\nPred FISTA w/ Line Search:", x_fista_ls)
print("FISTA function value w/ Line Search:", robust_huber(x_fista_ls, A, b, gamma, lam))

import matplotlib.pyplot as plt

def plot_losses_per_experiment(
    loss_data, 
    output_dir, 
    log_scale=False, 
    log_y_scale=False, 
    extend_converged=True, 
    offset=0.0
):
    """
    Plot and save loss data for multiple experiments.
    
    Parameters:
        loss_data (list[dict]): A list of dictionaries where each dictionary maps algorithm names to loss values.
        output_dir (str): Directory to save the plots.
        log_scale (bool): Whether to use a log scale for the x-axis.
        log_y_scale (bool): Whether to use a log scale for the y-axis.
        extend_converged (bool): Whether to extend the curve horizontally after convergence.
        offset (float): Offset for vertical lines when algorithms converge at the same step.
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'black']
    line_offsets = offset  # Offset for vertical lines

    for i, experiment in enumerate(loss_data):
        plt.figure(figsize=(10, 6))
        max_iters = max(len(vals) for vals in experiment.values())

        # Collect convergence points
        convergence_points = {}

        for j, (algorithm, losses) in enumerate(experiment.items()):
            color = colors[j % len(colors)]
            iters = np.arange(1, len(losses) + 1)

            # Plot the losses for the algorithm
            plt.plot(iters, losses, label=algorithm, color=color)

            # Add a vertical line at the convergence point
            convergence_iter = len(losses)
            if convergence_iter in convergence_points:
                convergence_points[convergence_iter] += 1
            else:
                convergence_points[convergence_iter] = 1

            # Offset for vertical line to avoid overlap
            vertical_offset = line_offsets * convergence_points[convergence_iter]
            plt.axvline(x=convergence_iter + vertical_offset, linestyle='--', color=color)

            # Optionally extend the curve horizontally after convergence
            if extend_converged and len(losses) < max_iters:
                plt.plot(
                    np.arange(len(losses) + 1, max_iters + 1),
                    [losses[-1]] * (max_iters - len(losses)),
                    color=color,
                    linestyle='-',
                )

        # Customize the plot
        plt.xlabel("Iterations (log scale)" if log_scale else "Iterations")
        plt.ylabel("Objective Value (log scale)" if log_y_scale else "Objective Value")
        plt.title(f"Experiment {i + 1} Losses")
        if log_scale:
            plt.xscale('log')
        if log_y_scale:
            plt.yscale('log')
        plt.legend()

        # Save the plot
        output_path = os.path.join(output_dir, f"experiment_{i + 1}_losses.png")
        plt.savefig(output_path)
        plt.close()

# Plot and save results
loss_data = [experiment_1_losses, experiment_2_losses, experiment_3_losses]
loss_data_2 = [experiment_1_losses_2, experiment_2_losses_2, experiment_3_losses_2]

plot_losses_per_experiment(loss_data, "plots/part1_py", log_scale=False, log_y_scale=False, extend_converged=False)
plot_losses_per_experiment(loss_data_2, "plots/part2_py", log_scale=False, log_y_scale=False, extend_converged=False, offset=0.0)
