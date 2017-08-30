# Phenotype functions
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

# linear reaction norm
function linear_norm(genotype::AbstractArray{Float64,1}, env::Array{Float64,1}, σ::Float64=0.1)
    return [genotype ⋅ env + rand(Normal(0,σ))]
end
