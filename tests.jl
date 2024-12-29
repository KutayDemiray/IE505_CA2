using Convex, SCS, LinearAlgebra, Distributions
include("hista.jl")

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

x_hista = HISTA(A, b, lambda, gamma)

println("Test 1")
println("True:", x_true)
println("Pred:", x_hista)
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

x_hista = HISTA(A, b, lambda, gamma)

println("Test 2")
println("True:", x_true)
println("Pred:", x_hista)
println("")

# Test 3

A = zeros(100, 10)
# we interpreted "sparse columns" as 10% of the rows in each column nonzero
for j in 1:10
    nonzero_indices = sample(1:100, 10, replace=false)
    A[nonzero_indices, j] .= randn(10)
end

# add outliers to A
outlier_columns = sample(1:10, 2, replace=false)
for col in outlier_columns
    outlier_indices = sample(1:100, 5, replace=false)
    A[outlier_indices, col] .= rand(Uniform(50, 100), 5)
end

x_true = zeros(10)
nonzero_indices = sample(1:10, 3, replace=false)
x_true[nonzero_indices] .= rand(Uniform(-10, 10), 3)

epsilon = rand(Normal(0, sigma^2), 100)
b = A * x_true + epsilon

x_hista = HISTA(A, b, lambda, gamma, 100000)

println("Test 3")
println("True:", x_true)
println("Pred:", x_hista)
println("")
