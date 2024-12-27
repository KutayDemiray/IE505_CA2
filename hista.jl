using Convex, SCS, LinearAlgebra
using Random, Distributions

A = randn(100, 10)

function huber_loss(z, gamma)
    if abs(z) <= gamma begin
        return z^2 / (2 * gamma)
    end
    else
        return abs(z) - gamma / 2
    end
end

function huber_derivative(z, gamma)
    if abs(z) <= gamma
        return z / gamma
    elseif z > 0
        return 1
    else
        return -1
    end
end

function huber_grad(z, gamma)
    grad = similar(z)
    for i in eachindex(grad)
        grad[i] = huber_derivative(z[i], gamma)
    end
    return grad
end

# proximal operator of the regularization term is the soft thresholding operator as described in the instructions
function prox_l1(z, lambda)
    prox = similar(z)
    for i in eachindex(z)
        prox[i] = sign(z[i]) * max(abs(z[i]) - lambda, 0)
    end
    return prox
end

function compute_lipschitz(A, gamma)    
    # lipschitz constant of an affine function
    # is the largest singular value of A
    sigma_max = svd(A).S[1]
    # for the overall function L_huber * L_aff^2
    return (sigma_max^2) / gamma
end

function HISTA(A, b, lambda, gamma, max_iter = 1000, tol = 1e-6)
    # Initialize variables
    x = zeros(size(A, 2))
    L = compute_lipschitz(A, gamma)
    eta = 1.0 / L  # constant step size

    for k in 1:max_iter
    # Gradient of the smooth part
    grad = A' * huber_grad(A * x - b, gamma)

    # Gradient descent step
    y = x - eta * grad

    # Proximal step
    x_new = prox_l1(y, lambda * eta)

    # Convergence check
    if norm(x_new - x) < tol
        println("Converged in $k iterations.")
        return x_new
    end

    x = x_new
    end

    println("Reached maximum iterations without full convergence.")
    return x
end
