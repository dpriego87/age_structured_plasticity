# Phenotype functions
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

function linear_norm(genotype::Array{Float64,1}, env::Array{Float64,1}, σ::Float64=0.1)
   return genotype ⋅ env + rand(Normal(0,σ))
end
