# Fitness functions
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

function gauss_purify_linear_norm(phenotype::Array{Float64,1}, env::Array{Float64,1},
                                  x::Array{Float64,1}, γ::Float64=1.0, wmax::Float64=5.0)
    opt = x ⋅ env
    return wmax * exp(- γ * norm(phenotype[1] - opt) / 2)
end
