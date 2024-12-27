using Convex, SCS, LinearAlgebra, Distributions
include("hista.jl")

# Task 1
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

x_hista = HISTA(A, b, lambda, gamma)

println(x_true)
println(x_hista)

