using Convex, SCS, LinearAlgebra
using Random, Distributions

A = randn(100, 10)

function huber_loss(z, gamma)
    if abs(z) <= gamma
        begin
            return z^2 / (2 * gamma)
        end
    else
        return abs(z) - gamma / 2
    end
end

function robust_huber(x, A, b, gamma, lambda)
    huber_part = sum(huber_loss.(A * x - b, gamma))
    l1_part = lambda * sum(abs.(x))
    return huber_part + l1_part
end

function robust_l2(x, A, b, lambda)
    l2_part = sum((A * x - b)' * (A * x - b)) / 2
    l1_part = lambda * sum(abs.(x))
    return l2_part + l1_part
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

function compute_lipschitz_huber(A, gamma)
    # lipschitz constant of an affine function
    # is the largest singular value of A
    sigma_max = svd(A).S[1]
    # for the overall function L_huber * L_aff^2
    return (sigma_max^2) / gamma
end

function HISTA(A, b, lambda, gamma, line_search=false, max_iter=100000, tol=1e-6, beta=0.9)
    # Initialize variables
    x = zeros(size(A, 2))
    x_new = copy(x)
    if line_search
        eta = 1.0
    else
        L = compute_lipschitz_huber(A, gamma)
        eta = 1.0 / L  # constant step size
    end

    obj_vals = Float64[]

    for k in 1:max_iter
        # Gradient of the smooth part
        grad = A' * huber_grad(A * x - b, gamma)

        # Backtracking line search
        if line_search
            eta_tmp = eta
            while true
                x_new = prox_l1(x - eta_tmp * grad, lambda * eta_tmp)
                G_eta = (x - x_new) / eta_tmp

                huber_current = sum(huber_loss.(A * x - b, gamma))
                huber_new = sum(huber_loss.(A * x_new - b, gamma))

                lhs = huber_new
                rhs = huber_current - eta_tmp * dot(grad, G_eta) + (eta_tmp / 2) * (norm(G_eta)^2)

                if lhs <= rhs
                    break
                end
                eta_tmp *= beta
            end
            eta = eta_tmp
        else
            # Gradient descent step
            y = x - eta * grad

            # Proximal step
            x_new = prox_l1(y, lambda * eta)
        end
        push!(obj_vals, robust_huber(x_new, A, b, gamma, lambda))

        #println("Iteration $k: $(norm(x_new - x))")
        # Convergence check
        if norm(x_new - x) < tol
            println("Converged in $k iterations.")
            return x_new, obj_vals
        end
        x = x_new
    end

    println("Reached maximum iterations without full convergence.")
    return x, obj_vals
end

function FastHISTA(A, b, lambda, gamma, line_search=false, init_t_with_L=false, max_iter=100000, tol=1e-6, beta=0.9)
    # Initialize variables
    x = zeros(size(A, 2))
    x_new = copy(x)
    x_prev = zeros(size(A, 2))
    if line_search
        if init_t_with_L
            L = compute_lipschitz_huber(A, gamma)
            eta = 1.0 / L
        else
            eta = 1.0
        end
    else
        L = compute_lipschitz_huber(A, gamma)
        eta = 1.0 / L  # constant step size
    end

    t = 1.0
    t_prev = 1.0
    if init_t_with_L
        L = compute_lipschitz_huber(A, gamma)
        t = 1 / L  # momentum parameter
        t_prev = 1 / L
    end

    obj_vals = Float64[]

    for k in 1:max_iter
        # Momentum update
        #momentum = min(1.0, (t_prev - 1) / t)
        momentum = (t_prev - 1) / t
        y = x + momentum * (x - x_prev)

        # Gradient of the smooth part
        grad = A' * huber_grad(A * y - b, gamma)

        # Backtracking line search
        if line_search
            eta_tmp = eta
            while true
                x_new = prox_l1(y - eta_tmp * grad, lambda * eta_tmp)
                G_eta = (y - x_new) / eta_tmp

                huber_current = sum(huber_loss.(A * y - b, gamma))
                huber_new = sum(huber_loss.(A * x_new - b, gamma))

                lhs = huber_new
                rhs = huber_current - eta_tmp * dot(grad, G_eta) + (eta_tmp / 2) * (norm(G_eta)^2)

                if lhs <= rhs
                    break
                end
                eta_tmp = max(eta_tmp * beta, 1e-6)  # Prevent shrinking too small
            end
            eta = eta_tmp
        else
            # Gradient descent step
            y = y - eta * grad

            # Proximal step
            x_new = prox_l1(y, lambda * eta)
        end

        # Update momentum parameter
        t_new = (1 + sqrt(1 + 4 * t^2)) / 2

        push!(obj_vals, robust_huber(x_new, A, b, gamma, lambda))
        #println("Iteration $k: $(norm(x_new - x))")
        # Convergence check
        if norm(x_new - x) < tol
            println("Converged in $k iterations.")
            return x_new, obj_vals
        end

        x_prev = x
        x = x_new
        t_prev = t
        t = t_new
    end

    println("Reached maximum iterations without full convergence.")
    return x, obj_vals
end


function ProxNewton(A, b, lambda, gamma, line_search=false, max_iter=100000, tol=1e-6, alpha=0.1, beta=0.5)
    x = zeros(size(A, 2))
    x_new = copy(x)

    L = compute_lipschitz_huber(A, gamma)

    epsilon = 1e-6  # Regularization parameter

    obj_vals = Float64[]

    for k in 1:max_iter
        # Compute residual
        residual = A * x - b

        # Compute gradient of the smooth part (Huber loss)
        grad = A' * huber_grad(residual, gamma)

        # Compute exact Hessian with regularization
        W = Diagonal([abs(residual[i]) <= gamma ? 1.0 / gamma : 0.0 for i in 1:length(residual)])
        H = A' * W * A + epsilon * I(size(A, 2))  # Regularized Hessian with added term to prevent singularity

        # Initialize `z` with a fallback value
        z = grad  # Fallback to gradient descent if no solution for z is found

        # Solve the Newton step (scaled proximal mapping)
        try
            z = H \ grad  # Solve linear system
        catch e
            if isa(e, SingularException)
                println("Warning: Singular Hessian detected, using pseudoinverse.")
                z = pinv(H) * grad  # Use pseudoinverse as a fallback
            else
                rethrow(e)
            end
        end

        # Compute proximal mapping and direction
        prox_x = prox_l1(x - z, lambda)
        v = prox_x - x  # Direction

        # Backtracking line search
        t = 1.0  # Initial step size
        while line_search
            x_trial = x + t * v
            f_x = robust_huber(x, A, b, gamma, lambda)
            f_x_trial = robust_huber(x_trial, A, b, gamma, lambda)
            h_x = lambda * sum(abs.(x))
            h_x_trial = lambda * sum(abs.(x_trial))

            lhs = f_x_trial
            rhs = f_x + alpha * t * dot(grad, v) + alpha * (h_x_trial - h_x)

            if lhs <= rhs
                break
            end
            t *= beta  # Shrink step size
            if t < 1e-12
                println("Step size too small, exiting line search.")
                break
            end
        end

        # Update solution
        x_new = x + t * v

        if !line_search
            x_new = x + (1 / L) * v
        end

        push!(obj_vals, robust_huber(x_new, A, b, gamma, lambda))

        # Convergence check
        if norm(x_new - x) < tol
            println("Converged in $k iterations.")
            return x_new, obj_vals
        end

        x = x_new
    end

    println("Reached maximum iterations without full convergence.")
    return x, obj_vals
end

function compute_lipschitz_l2(A)
    # Lipschitz constant for the gradient of the L2 loss
    sigma_max = svd(A).S[1]  # Largest singular value of A
    return sigma_max^2
end

function FISTA(A, b, lambda, line_search=false, max_iter=100000, tol=1e-6, beta=0.9)
    # Initialize variables
    x = zeros(size(A, 2))
    x_new = copy(x)
    x_prev = zeros(size(A, 2))
    if line_search
        eta = 1.0
    else
        L = compute_lipschitz_l2(A)  # Use the separate function for Lipschitz computation
        eta = 1.0 / L  # constant step size
    end
    t = 1.0  # momentum parameter
    t_prev = 1.0

    obj_vals = Float64[]

    for k in 1:max_iter
        # Momentum update
        #momentum = min(1.0, (t_prev - 1) / t)
        momentum = (t_prev - 1) / t
        y = x + momentum * (x - x_prev)

        # Gradient of the smooth part (L2 loss)
        grad = A' * (A * y - b)

        # Backtracking line search
        if line_search
            eta_tmp = eta
            while true
                x_new = prox_l1(y - eta_tmp * grad, lambda * eta_tmp)
                G_eta = (y - x_new) / eta_tmp

                l2_current = 0.5 * norm(A * y - b)^2
                l2_new = 0.5 * norm(A * x_new - b)^2

                lhs = l2_new
                rhs = l2_current - eta_tmp * dot(grad, G_eta) + (eta_tmp / 2) * (norm(G_eta)^2)

                if lhs <= rhs
                    break
                end
                eta_tmp = max(eta_tmp * beta, 1e-6)  # Prevent shrinking too small
            end
            eta = eta_tmp
        else
            # Gradient descent step
            y = y - eta * grad

            # Proximal step
            x_new = prox_l1(y, lambda * eta)
        end

        # Update momentum parameter
        t_new = (1 + sqrt(1 + 4 * t^2)) / 2

        push!(obj_vals, robust_l2(x_new, A, b, lambda))

        # Convergence check
        if norm(x_new - x) < tol
            println("Converged in $k iterations.")
            return x_new, obj_vals
        end

        x_prev = x
        x = x_new
        t_prev = t
        t = t_new
    end

    println("Reached maximum iterations without full convergence.")
    return x, obj_vals
end
