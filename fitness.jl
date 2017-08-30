# Fitness functions
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

function gauss_purify_linear_norm(genotype::AbstractArray{Float64,1},
                                  phenotype::Array{Float64,1},
                                  env::Array{Float64,1},
                                  x::Array{Float64,1}, γ::Float64=1.0, wmax::Float64=5.0)
    opt = x ⋅ env
    return wmax * exp(- γ * (phenotype[1] - opt)^2 / 2)
end

# gaussian fitness function with cost of plasticity
function gauss_purify_cost_linear_norm(genotype::AbstractArray{Float64,1},
                                       phenotype::Array{Float64,1},
                                       env::Array{Float64,1},
                                       x::Array{Float64,1}, γ::Float64=1.0, γb::Float64=1.0,
                                       wmax::Float64=5.0)
    opt = x ⋅ env
    return wmax * exp(- γ * (phenotype[1] - opt)^2 / 2 - γb * genotype[2]^2 / 2)
end

