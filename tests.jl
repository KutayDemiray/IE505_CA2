using Convex, SCS, LinearAlgebra, Distributions
include("hista.jl")
include("plot.jl")

seed = 10
Random.seed!(seed)

# Test 1
sigma = 1
gamma = 1
lambda = 0.1

A = randn(100, 10)
x_true = zeros(10)
nonzero_indices = sample(1:10, 3, replace=false)
# i am not sure how we are supposed to initialize the nonzero entries
# so i sampled some random values uniformly
x_true[nonzero_indices] .= rand(Uniform(-10, 10), 3)
epsilon = rand(Normal(0, sigma^2), (100, 1))

b = A * x_true + epsilon
outlier_indices = sample(1:100, 5, replace=false)
b[outlier_indices] .= rand(Uniform(50, 100), 5)

x_hista, x_hista_vals = HISTA(A, b, lambda, gamma)
x_hista_ls, x_hista_ls_vals = HISTA(A, b, lambda, gamma, true)
x_fasthista, x_fasthista_vals = FastHISTA(A, b, lambda, gamma)
x_fasthista_ls, x_fasthista_ls_vals = FastHISTA(A, b, lambda, gamma, true)
x_proxnewton, x_proxnewton_vals = ProxNewton(A, b, lambda, gamma)
x_proxnewton_ls, x_proxnewton_ls_vals = ProxNewton(A, b, lambda, gamma, true)
x_fista, x_fista_vals = FISTA(A, b, lambda)
x_fista_ls, x_fista_ls_vals = FISTA(A, b, lambda, true)

experiment_1_losses = Dict(
    "HISTA" => x_hista_vals,
    "HISTA_LS" => x_hista_ls_vals,
    "FastHISTA" => x_fasthista_vals,
    "FastHISTA_LS" => x_fasthista_ls_vals,
    #"ProxNewton" => x_proxnewton_vals,
    #"ProxNewton_LS" => x_proxnewton_ls_vals,
    #"FISTA" => x_fista_vals,
    #"FISTA_LS" => x_fista_ls_vals
)

println("Test 1")
println("True:", x_true)
println("True function value:", robust_huber(x_true, A, b, gamma, lambda))
println()
println("Pred HISTA:", x_hista)
println("HISTA function value:", robust_huber(x_hista, A, b, gamma, lambda))
println()
println("Pred HISTA w/ Line Search:", x_hista_ls)
println("HISTA function value w/ Line Search", robust_huber(x_hista_ls, A, b, gamma, lambda))
println()
println("Pred FastHISTA:", x_fasthista)
println("FastHISTA function value:", robust_huber(x_fasthista, A, b, gamma, lambda))
println()
println("Pred FastHISTA w/ Line Search:", x_fasthista_ls)
println("FastHISTA function value w/ Line Search:", robust_huber(x_fasthista_ls, A, b, gamma, lambda))
println()
println("Pred ProxNewton:", x_proxnewton)
println("ProxNewton function value:", robust_huber(x_proxnewton, A, b, gamma, lambda))
println()
println("Pred ProxNewton w/ Line Search:", x_proxnewton_ls)
println("ProxNewton function value w/ Line Search:", robust_huber(x_proxnewton_ls, A, b, gamma, lambda))
println()
println("Pred FISTA:", x_fista)
println("FISTA function value:", robust_huber(x_fista, A, b, gamma, lambda))
println()
println("Pred FISTA w/ Line Search:", x_fista_ls)
println("FISTA function value w/ Line Search:", robust_huber(x_fista_ls, A, b, gamma, lambda))
println("")

# Test 2
A = randn(200, 20)

x_true = zeros(20)
nonzero_indices = sample(1:20, 3, replace=false)
# i am not sure how we are supposed to initialize the nonzero entries
# so i sampled some random values uniformly
x_true[nonzero_indices] .= rand(Uniform(-10, 10), 3)

epsilon = rand(Normal(0, 0.1^2), (200, 1))
# a subset of the epsilons should have higher variance
outlier_indices = sample(1:100, 10, replace=false)
epsilon[outlier_indices] .= rand(Normal(0, 5^2), 10)

b = A * x_true + epsilon

x_hista, x_hista_vals = HISTA(A, b, lambda, gamma)
x_hista_ls, x_hista_ls_vals = HISTA(A, b, lambda, gamma, true)
x_fasthista, x_fasthista_vals = FastHISTA(A, b, lambda, gamma)
x_fasthista_ls, x_fasthista_ls_vals = FastHISTA(A, b, lambda, gamma, true)
x_proxnewton, x_proxnewton_vals = ProxNewton(A, b, lambda, gamma)
x_proxnewton_ls, x_proxnewton_ls_vals = ProxNewton(A, b, lambda, gamma, true)
x_fista, x_fista_vals = FISTA(A, b, lambda)
x_fista_ls, x_fista_ls_vals = FISTA(A, b, lambda, true)

experiment_2_losses = Dict(
    "HISTA" => x_hista_vals,
    "HISTA_LS" => x_hista_ls_vals,
    "FastHISTA" => x_fasthista_vals,
    "FastHISTA_LS" => x_fasthista_ls_vals,
    #"ProxNewton" => x_proxnewton_vals,
    #"ProxNewton_LS" => x_proxnewton_ls_vals,
    #"FISTA" => x_fista_vals,
    #"FISTA_LS" => x_fista_ls_vals
)

println("Test 2")
println("True:", x_true)
println("True function value:", robust_huber(x_true, A, b, gamma, lambda))
println()
println("Pred HISTA:", x_hista)
println("HISTA function value:", robust_huber(x_hista, A, b, gamma, lambda))
println()
println("Pred HISTA w/ Line Search:", x_hista_ls)
println("HISTA function value w/ Line Search", robust_huber(x_hista_ls, A, b, gamma, lambda))
println()
println("Pred FastHISTA:", x_fasthista)
println("FastHISTA function value:", robust_huber(x_fasthista, A, b, gamma, lambda))
println()
println("Pred FastHISTA w/ Line Search:", x_fasthista_ls)
println("FastHISTA function value w/ Line Search:", robust_huber(x_fasthista_ls, A, b, gamma, lambda))
println()
println("Pred ProxNewton:", x_proxnewton)
println("ProxNewton function value:", robust_huber(x_proxnewton, A, b, gamma, lambda))
println()
println("Pred ProxNewton w/ Line Search:", x_proxnewton_ls)
println("ProxNewton function value w/ Line Search:", robust_huber(x_proxnewton_ls, A, b, gamma, lambda))
println()
println("Pred FISTA:", x_fista)
println("FISTA function value:", robust_huber(x_fista, A, b, gamma, lambda))
println()
println("Pred FISTA w/ Line Search:", x_fista_ls)
println("FISTA function value w/ Line Search:", robust_huber(x_fista_ls, A, b, gamma, lambda))
println("")

# Test 3

A = zeros(100, 10)
# we interpreted "sparse columns" as 10% of the rows in each column nonzero
for j in 1:10
    local nonzero_indices = sample(1:100, 10, replace=false)
    A[nonzero_indices, j] .= randn(10)
end

# add outliers to A
outlier_columns = sample(1:10, 2, replace=false)
for col in outlier_columns
    local outlier_indices = sample(1:100, 5, replace=false)
    A[outlier_indices, col] .= rand(Uniform(50, 100), 5)
end

x_true = zeros(10)
nonzero_indices = sample(1:10, 3, replace=false)
x_true[nonzero_indices] .= rand(Uniform(-10, 10), 3)

epsilon = rand(Normal(0, sigma^2), 100)
b = A * x_true + epsilon

x_hista, x_hista_vals = HISTA(A, b, lambda, gamma)
x_hista_ls, x_hista_ls_vals = HISTA(A, b, lambda, gamma, true)
x_fasthista, x_fasthista_vals = FastHISTA(A, b, lambda, gamma)
x_fasthista_ls, x_fasthista_ls_vals = FastHISTA(A, b, lambda, gamma, true)
x_proxnewton, x_proxnewton_vals = ProxNewton(A, b, lambda, gamma)
x_proxnewton_ls, x_proxnewton_ls_vals = ProxNewton(A, b, lambda, gamma, true)
x_fista, x_fista_vals = FISTA(A, b, lambda)
x_fista_ls, x_fista_ls_vals = FISTA(A, b, lambda, true)

experiment_3_losses = Dict(
    "HISTA" => x_hista_vals,
    "HISTA_LS" => x_hista_ls_vals,
    "FastHISTA" => x_fasthista_vals,
    "FastHISTA_LS" => x_fasthista_ls_vals,
    #"ProxNewton" => x_proxnewton_vals,
    #"ProxNewton_LS" => x_proxnewton_ls_vals,
    #"FISTA" => x_fista_vals,
    #"FISTA_LS" => x_fista_ls_vals
)

println("Test 3")
println("True:", x_true)
println("True function value:", robust_huber(x_true, A, b, gamma, lambda))
println()
println("Pred HISTA:", x_hista)
println("HISTA function value:", robust_huber(x_hista, A, b, gamma, lambda))
println()
println("Pred HISTA w/ Line Search:", x_hista_ls)
println("HISTA function value w/ Line Search", robust_huber(x_hista_ls, A, b, gamma, lambda))
println()
println("Pred FastHISTA:", x_fasthista)
println("FastHISTA function value:", robust_huber(x_fasthista, A, b, gamma, lambda))
println()
println("Pred FastHISTA w/ Line Search:", x_fasthista_ls)
println("FastHISTA function value w/ Line Search:", robust_huber(x_fasthista_ls, A, b, gamma, lambda))
println()
println("Pred ProxNewton:", x_proxnewton)
println("ProxNewton function value:", robust_huber(x_proxnewton, A, b, gamma, lambda))
println()
println("Pred ProxNewton w/ Line Search:", x_proxnewton_ls)
println("ProxNewton function value w/ Line Search:", robust_huber(x_proxnewton_ls, A, b, gamma, lambda))
println()
println("Pred FISTA:", x_fista)
println("FISTA function value:", robust_huber(x_fista, A, b, gamma, lambda))
println()
println("Pred FISTA w/ Line Search:", x_fista_ls)
println("FISTA function value w/ Line Search:", robust_huber(x_fista_ls, A, b, gamma, lambda))
println("")

# plot losses
loss_data = [experiment_1_losses, experiment_2_losses, experiment_3_losses]

plot_losses_per_experiment(loss_data, "plots/part1", false, false, false)
