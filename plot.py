import os
import numpy as np
import matplotlib.pyplot as plt


def plot_losses_per_experiment(
    loss_data, 
    output_dir, 
    log_scale=False, 
    log_y_scale=False, 
    extend_converged=True, 
    offset=0.2
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
