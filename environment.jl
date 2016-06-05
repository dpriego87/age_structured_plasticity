# Environment update functions
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

using Distributions

# "autoregressive" type model
function autoregressive(x::Array{Float64,1}, θ::Array{Float64,1}=[0.5], σ::Float64=1.0)
    # autoregressive model with Gaussian noise
    x_new = x ⋅ θ + rand(Normal(0,σ))
    return [x_new; x[2:end]]
end
