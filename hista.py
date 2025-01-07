import numpy as np
from tqdm import tqdm


def huber_loss(z, gamma):
    return np.where(np.abs(z) <= gamma, z**2 / (2 * gamma), np.abs(z) - gamma / 2)


def robust_huber(x, A, b, gamma, lam):
    huber_part = np.sum(huber_loss(A @ x - b, gamma))
    l1_part = lam * np.sum(np.abs(x))
    return huber_part + l1_part


def robust_l2(x, A, b, lam):
    l2_part = np.sum((A @ x - b).T @ (A @ x - b)) / 2
    l1_part = lam * np.sum(np.abs(x))
    return l2_part + l1_part


def huber_derivative(z, gamma):
    return np.where(np.abs(z) <= gamma, z / gamma, np.sign(z))


def huber_grad(z, gamma):
    return huber_derivative(z, gamma)


def prox_l1(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)


def compute_lipschitz_huber(A, gamma):
    sigma_max = np.linalg.svd(A, compute_uv=False)[0]
    return sigma_max**2 / gamma


def compute_lipschitz_l2(A):
    sigma_max = np.linalg.svd(A, compute_uv=False)[0]
    return sigma_max**2


def HISTA(A, b, lam, gamma, line_search=False, max_iter=100000, tol=1e-6, beta=0.9):
    x = np.zeros_like(b)
    eta = 1.0 / compute_lipschitz_huber(A, gamma) if not line_search else 1.0
    obj_vals = []

    for k in tqdm(range(max_iter)):
        grad = A.T @ huber_grad(A @ x - b, gamma)
        if line_search:
            eta_tmp = eta
            while True:
                x_new = prox_l1(x - eta_tmp * grad, lam * eta_tmp)
                G_eta = (x - x_new) / eta_tmp
                lhs = np.sum(huber_loss(A @ x_new - b, gamma))
                rhs = (
                    np.sum(huber_loss(A @ x - b, gamma))
                    - eta_tmp * np.dot(grad, G_eta)
                    + eta_tmp / 2 * np.linalg.norm(G_eta) ** 2
                )
                if lhs <= rhs:
                    break
                eta_tmp *= beta
            eta = eta_tmp
        else:
            y = x - eta * grad
            x_new = prox_l1(y, lam * eta)

        obj_vals.append(robust_huber(x_new, A, b, gamma, lam))
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {k + 1} iterations.")
            return x_new, obj_vals
        x = x_new

    print("Reached maximum iterations without full convergence.")
    return x, obj_vals


def FastHISTA(
    A,
    b,
    lam,
    gamma,
    line_search=False,
    init_t_with_L=False,
    max_iter=100000,
    tol=1e-6,
    beta=0.9,
):
    x = np.zeros_like(b)
    x_prev = np.zeros_like(x)
    t = t_prev = 1 / compute_lipschitz_huber(A, gamma) if init_t_with_L else 1.0
    eta = 1.0 / compute_lipschitz_huber(A, gamma) if not line_search else 1.0
    obj_vals = []

    for k in tqdm(range(max_iter)):
        momentum = (t_prev - 1) / t
        y = x + momentum * (x - x_prev)
        grad = A.T @ huber_grad(A @ y - b, gamma)
        if line_search:
            eta_tmp = eta
            while True:
                x_new = prox_l1(y - eta_tmp * grad, lam * eta_tmp)
                G_eta = (y - x_new) / eta_tmp
                lhs = np.sum(huber_loss(A @ x_new - b, gamma))
                rhs = (
                    np.sum(huber_loss(A @ y - b, gamma))
                    - eta_tmp * np.dot(grad, G_eta)
                    + eta_tmp / 2 * np.linalg.norm(G_eta) ** 2
                )
                if lhs <= rhs:
                    break
                eta_tmp = max(eta_tmp * beta, 1e-6)
            eta = eta_tmp
        else:
            y = y - eta * grad
            x_new = prox_l1(y, lam * eta)

        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        obj_vals.append(robust_huber(x_new, A, b, gamma, lam))
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {k + 1} iterations.")
            return x_new, obj_vals
        x_prev, x, t_prev, t = x, x_new, t, t_new

    print("Reached maximum iterations without full convergence.")
    return x, obj_vals


def ProxNewton(
    A, b, lam, gamma, line_search=False, max_iter=100000, tol=1e-6, alpha=0.1, beta=0.5
):
    x = np.zeros_like(b)
    L = compute_lipschitz_huber(A, gamma)
    epsilon = 1e-6  # Regularization parameter
    obj_vals = []

    for k in tqdm(range(max_iter)):
        residual = A @ x - b
        grad = A.T @ huber_grad(residual, gamma)
        W = np.diag(
            [
                1.0 / gamma if np.abs(residual[i]) <= gamma else 0.0
                for i in range(len(residual))
            ]
        )
        H = A.T @ W @ A + epsilon * np.eye(A.shape[1])

        try:
            z = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            print("Warning: Singular Hessian detected, using pseudoinverse.")
            z = np.linalg.pinv(H) @ grad

        prox_x = prox_l1(x - z, lam)
        v = prox_x - x
        t = 1.0

        if line_search:
            while True:
                x_new = x + t * v
                f_x = robust_huber(x, A, b, gamma, lam)
                f_x_new = robust_huber(x_new, A, b, gamma, lam)
                h_x = lam * np.sum(np.abs(x))
                h_x_new = lam * np.sum(np.abs(x_new))

                lhs = f_x_new
                rhs = f_x + alpha * t * np.dot(grad, v) + alpha * (h_x_new - h_x)

                if lhs <= rhs:
                    break
                t *= beta
                if t < 1e-12:
                    print("Step size too small, exiting line search.")
                    break
        else:
            x_new = x + (1 / L) * v

        obj_vals.append(robust_huber(x_new, A, b, gamma, lam))

        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {k + 1} iterations.")
            return x_new, obj_vals

        x = x_new

    print("Reached maximum iterations without full convergence.")
    return x, obj_vals


def FISTA(A, b, lam, line_search=False, max_iter=100000, tol=1e-6, beta=0.9):
    x = np.zeros_like(b)
    x_prev = np.zeros_like(x)
    eta = 1.0 / compute_lipschitz_l2(A) if not line_search else 1.0
    t = t_prev = 1.0
    obj_vals = []

    for k in tqdm(range(max_iter)):
        momentum = (t_prev - 1) / t
        y = x + momentum * (x - x_prev)
        grad = A.T @ (A @ y - b)

        if line_search:
            eta_tmp = eta
            while True:
                x_new = prox_l1(y - eta_tmp * grad, lam * eta_tmp)
                G_eta = (y - x_new) / eta_tmp
                l2_current = 0.5 * np.linalg.norm(A @ y - b) ** 2
                l2_new = 0.5 * np.linalg.norm(A @ x_new - b) ** 2

                lhs = l2_new
                rhs = (
                    l2_current
                    - eta_tmp * np.dot(grad, G_eta)
                    + (eta_tmp / 2) * np.linalg.norm(G_eta) ** 2
                )

                if lhs <= rhs:
                    break
                eta_tmp = max(eta_tmp * beta, 1e-6)
            eta = eta_tmp
        else:
            y = y - eta * grad
            x_new = prox_l1(y, lam * eta)

        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        obj_vals.append(robust_l2(x_new, A, b, lam))

        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {k + 1} iterations.")
            return x_new, obj_vals

        x_prev, x, t_prev, t = x, x_new, t, t_new

    print("Reached maximum iterations without full convergence.")
    return x, obj_vals
