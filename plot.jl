using Plots

function plot_losses_per_experiment(
    loss_data::Vector{Dict{String, Vector{Float64}}}, 
    output_dir::String, 
    log_scale::Bool=false, 
    log_y_scale::Bool=false, 
    extend_converged::Bool=true,
    offset::Float64=0.2
)
    mkpath(output_dir)

    colors = [:blue, :green, :red, :orange, :purple, :cyan, :magenta, :black]
    line_offsets = offset  # Offset for vertical lines when algorithms converge at the same step

    for (i, experiment) in enumerate(loss_data)
        plot()  # Start a new plot for each experiment
        max_iters = maximum([length(vals) for vals in values(experiment)])
        
        # Collect convergence points
        convergence_points = Dict{Int, Int}()
        for (j, (algorithm, losses)) in enumerate(experiment)
            color = colors[j]
            iters = 1:length(losses)

            # Plot the losses for the algorithm
            plot!(iters, losses, label=algorithm, color=color)

            # Add a vertical line at the convergence point
            convergence_iter = length(losses)
            if haskey(convergence_points, convergence_iter)
                convergence_points[convergence_iter] += 1  # Increment count for this iteration
            else
                convergence_points[convergence_iter] = 1  # Initialize count
            end

            offset = line_offsets * convergence_points[convergence_iter]  # Offset for this iteration
            vline!([convergence_iter + offset], label="", linestyle=:dash, color=color)

            # Optionally extend the curve horizontally after convergence
            if extend_converged && length(losses) < max_iters
                plot!(length(losses):max_iters, fill(losses[end], max_iters - length(losses) + 1),
                      label="", color=color, linestyle=:solid)
            end
        end

        # Customize the plot
        xlabel!(log_scale ? "Iterations (log scale)" : "Iterations")
        ylabel!(log_y_scale ? "Objective Value (log scale)" : "Objective Value")
        title!("Experiment $i Losses")
        if log_scale
            xaxis!(:log10)
        end
        if log_y_scale
            yaxis!(:log10)
        end

        # Save the plot
        savefig(joinpath(output_dir, "experiment_$(i)_losses.png"))
    end
end
