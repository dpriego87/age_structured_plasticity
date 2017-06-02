## Plasticity and senescence simulation
##
## Jeremy Van Cleve <jeremy.vancleve@gmail.com>

include("populations.jl")
include("environment.jl")
include("phenotype.jl")
include("fitness.jl")

using Distributions
using populations

##
## Continuum of alleles, linear reaction norm with cost of slope.
## Test against theory from Lande (2014, JEB)
##

## Assume a linear reaction norm with no past time dependence where
## eqn 2c in Lande (2014) becomes
## z_t = a + e + b ε_t
## Then the predicted evolved slope from eqn 6b is given below
function landeSlope(A, B, γ, γb, venv, arθ)
    # variance for autoregressive process of order one
    v = venv / (1 - arθ^2)
    # predicted slopes from Lande (2014, JEB)
    return [A, B / (1 + γb / (γ * v))]
end

function runSim(p, reps, burns, iters, verbose=true)
    ngeno = length(p.members[1].genotype)
    nenv  = length(p.env_state)
    mgeno = zeros((ngeno, reps*iters))
    env   = zeros((nenv, reps*iters))

    totalruns = burns+iters
    iterblock = round(reps * totalruns / 100)
    
    for r = 1:reps
        # reset genotypes
        for i = 1:p.size
            p.members[i].genotype = zeros(ngeno)
        end

        for j = 1:totalruns
            next_gen(p)
            # record after burn in
            if j > burns
                mgeno[:,(r-1)*iters+j-burns] = mean_genotype(p)
                env[:,(r-1)*iters+j-burns]   = p.env_state
            end
            # print %done
            if ((r-1)*totalruns+j) % iterblock == 0 && verbose == true
                print(round(Int,100.*((r-1)*totalruns+j)/(totalruns*reps)), "% ")
            end
        end
    end

    return (mgeno, env)
end

function popLinearPlasticityFert(;n::Int64=100, s::Array{Float64,1}=[0.0], ve::Float64=0.01,
                                 A::Float64=1.0, B::Float64=1.0, γ::Float64=2.0, γb::Float64=0.0, wmax::Float64=1.0,
                                 venv::Float64=0.1, arθ::Float64=0.75, vmut::Float64=0.01)
    nages = length(s)
    stdenv = sqrt(venv)
    AB = [A, B]
    varθ = [arθ]

    function phenof(i::Individual, e::Array{Float64,1})
        copy!(i.phenotype, linear_norm_g(i.genotype[i.age*2+1:i.age*2+2], e, ve))
    end

    function fitf(i::Individual, e::Array{Float64,1})
        if i.age < nages
            i.fitness[1] = s[i.age+1]
        else
            i.fitness[1] = 0.0
        end
        i.fitness[2] = gauss_purify_cost_linear_norm(i.phenotype, e, AB, γ, γb, wmax)
    end

    function mutf(offspring::Individual, parent::Individual)
        copy!(offspring.genotype,
              parent.genotype + rand(Normal(0.0,vmut), size(parent.genotype)))
    end

    function envf(e::Array{Float64,1})
        env = zeros(2)
        env[1] = 1.0
        env[2] = autoregressive([e[2]], varθ, stdenv)[1]
        return env
    end

    p = Population(# population size
                   n,
                   # genotype->phenotype map
                   phenof, 3,
                   # fitness function
                   fitf,
                   # mutation function
                   mutf,
                   # initial genotype function
                   (i)->zeros(2*nages),
                   envf, # env update function
                   [1.0, 0.0]); # initial env state

    return p
end

function popLinearPlasticitySurv(;n::Int64=100, s::Array{Float64,1}=[0.0], ve::Float64=0.01,
                                 A::Float64=1.0, B::Float64=1.0, γ::Float64=2.0, γb::Float64=0.0, wmax::Float64=1.0,
                                 venv::Float64=0.1, arθ::Float64=0.75, vmut::Float64=0.01)
    nages = length(s)
    stdenv = sqrt(venv)
    AB = [A, B]
    varθ = [arθ]
    
    function phenof(i::Individual, e::Array{Float64,1})
        copy!(i.phenotype, linear_norm_g(i.genotype[i.age*2+1:i.age*2+2], e, ve))
    end

    function fitf(i::Individual, e)
        if i.age < nages - 1
            i.fitness[1] = s[i.age+1] +
            gauss_purify_cost_linear_norm(i.phenotype, e, AB, γ, γb, wmax)
        else
            i.fitness[1] = 0.0
        end
        i.fitness[2] = 1.0
    end

    function mutf(offspring::Individual, parent::Individual)
        copy!(offspring.genotype,
              parent.genotype + rand(Normal(0.0,vmut), size(parent.genotype)))
    end

    function envf(e::Array{Float64,1})
        env = zeros(2)
        env[1] = 1.0
        env[2] = autoregressive([e[2]], varθ, stdenv)[1]
        return env
    end

    p = Population(# population size
                   n,
                   # genotype->phenotype map
                   phenof, 3,
                   # fitness function
                   fitf,
                   # mutation function
                   mutf,
                   # initial genotype function
                   (i)->zeros(2*nages),
                   envf, # env update function
                   [1.0, 0.0]); # initial env state

    return p
end

function plotBoxPlot(vals, expect, xticklab, ycolor)
    gcf()[:set_size_inches](4,3)
    ax = gca()
    bp = boxplot(vals, sym="")
    setp(bp["boxes"], color="black")
    setp(bp["medians"], color=ycolor)
    setp(bp["whiskers"], color="black", dashes=(3,3))
    
    plot(0:size(vals)[2]+1, fill(expect,size(vals)[2]+2), linestyle="--", dashes=(5,5), color=ycolor)
    
    ax[:set_xticklabels](xticklab)
    ax[:set_ylim](-3.0,3.0)
    ax[:set_xlabel]("Age class")
    ax[:set_ylabel]("Genotypic value")
end
