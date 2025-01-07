using SparseArrays, LinearAlgebra, Images
include("hista.jl")  # Load the unmodified implementations from hista.jl

# Define Gaussian kernel
function gaussian_kernel(size, sigma)
    center = (size - 1) / 2
    x = -center:center
    y = -center:center
    kernel = exp.(-(x' .^ 2 .+ y .^ 2) / (2 * sigma^2))
    return kernel ./ sum(kernel)  # Normalize
end

# Construct sparse blurring matrix
function construct_blur_matrix(image_size, kernel)
    n_rows, n_cols = image_size
    total_pixels = n_rows * n_cols
    kernel_size = size(kernel)
    center = div(kernel_size[1], 2)

    # Sparse matrix construction
    rows = Int[]
    cols = Int[]
    values = Float64[]

    for r in 1:n_rows
        for c in 1:n_cols
            pixel_index = (r - 1) * n_cols + c
            for kr in 1:kernel_size[1]
                for kc in 1:kernel_size[2]
                    rr = r + kr - center - 1
                    cc = c + kc - center - 1
                    if rr >= 1 && rr <= n_rows && cc >= 1 && cc <= n_cols
                        neighbor_index = (rr - 1) * n_cols + cc
                        push!(rows, pixel_index)
                        push!(cols, neighbor_index)
                        push!(values, kernel[kr, kc])
                    end
                end
            end
        end
    end

    return sparse(rows, cols, values, total_pixels, total_pixels)
end

# Override huber_grad to use FFT
function huber_grad(z, gamma)
    #println("huber_grad start")
    grad = similar(z)
    for i in eachindex(grad)
        grad[i] = abs(z[i]) <= gamma ? z[i] / gamma : sign(z[i])
    end
    #println("huber_grad end")
    return grad
end

# Override compute_lipschitz_huber for sparse matrix computation
function compute_lipschitz_huber(A, gamma)
    println("compute_lipschitz_huber start")
    # Lipschitz constant approximation using power iteration
    x = rand(size(A, 2))
    for _ in 1:10  # Run power iteration
        x = A' * (A * x)
        x /= norm(x)
    end
    L = norm(A * x)^2 / gamma
    println("compute_lipschitz_huber end")
    return L
end

# Main deblurring workflow
function deblur_image(blurred_image, kernel, lambda, gamma, max_iter=500, tol=1e-6)
    # Image size
    image_size = size(blurred_image)

    # Construct the blurring operator as a sparse matrix
    println("Start deblur")
    A = construct_blur_matrix(image_size, kernel)
    println("Constructed blur matrix A")
    # Vectorize the blurred image
    b = reshape(blurred_image, :)

    # Compute Lipschitz constant
    L = compute_lipschitz_huber(A, gamma)
    eta = 1.0 / L  # Step size

    # Run HISTA
    println("Starting FHISTA")
    x_recovered, obj_vals = FastHISTA(A, b, lambda, gamma, true, false, max_iter, 1e-3)
    println("Finished FHISTA")

    # Reshape recovered vector into an image
    deblurred_image = reshape(x_recovered, image_size)
    return deblurred_image, obj_vals
end

# Example usage
function main()
    # Load and preprocess the image
    blurred_image = Float64.(Gray.(load("cameraman.jpg")))

    # Define Gaussian kernel
    kernel = gaussian_kernel(5, 1.0)

    # Regularization and Huber loss parameters
    lambda = 0.01
    gamma = 5.0

    # Perform deblurring
    deblurred_image, obj_vals = deblur_image(blurred_image, kernel, lambda, gamma)

    # Save the result
    save("output/deblurred_image_lambda$(lambda)_gamma$(gamma).jpg", deblurred_image)
    println("Deblurred image saved to output/deblurred_image_lambda$(lambda)_gamma$(gamma).jpg")
end

main()
