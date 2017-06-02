# Environment update functions
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

using Distributions

# "autoregressive" type model
function autoregressive(x::Array{Float64,1}, θ::Array{Float64,1}=[0.5], σ::Float64=1.0)
    x_new = zeros(size(x))
    # autoregressive model with Gaussian noise
    x_new[1] = x ⋅ θ + rand(Normal(0,σ))
    # previous environmental states
    x_new[2:end-1] = x[1:end-1]
    
    return x_new
end
