# Environment update functions
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

using Distributions

# "autoregressive" type model
function AR(x::Array{Float64,1}, θ::Array{Float64,1}=[0.5], c::Float64=0., σ::Float64=1.)
    p = length(x)
    # autoregressive model with Gaussian noise
    x_new = c + x⋅θ + rand(Normal(0,σ))
    return [x_new; x[2:end]]
end
