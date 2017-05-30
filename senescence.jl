## Plasticity and senescence simulation
##
## Jeremy Van Cleve <jeremy.vancleve@gmail.com>

include("populations.jl")
include("environment.jl")
include("phenotype.jl")
include("fitness.jl")

using Distributions
using PyPlot
using JLD
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

function runSim(p, reps, burns, iters)
    ngeno = length(p.members[1].genotype)
    nenv  = length(p.env_state)
    mgeno = zeros((ngeno, reps*iters))
    env   = zeros((nenv, reps*iters))

    for r = 1:reps
        # reset genotypes
        for i = 1:p.size
            p.members[i].genotype = zeros(ngeno)
        end
        # burn-in
        for j = 1:burns
            next_gen(p)
        end
        # main iteration
        for j = 1:iters
            next_gen(p)
            mgeno[:,(r-1)*iters+j] = mean_genotype(p)
            env[:,(r-1)*iters+j]   = p.env_state
        end
        if r % round(reps / 100) == 0
            print(round(Int,100*r/reps), "% ")
        end
    end

    return (mgeno, env)
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

function linearPlasticityFertAgePop(n, s, ve, A, B, γ, γb, wmax, venv, arθ, vmut)
    nages = length(s)
    stdenv = sqrt(venv)

    function phenof(i::Individual, e)
        copy!(i.phenotype, linear_norm_g(i.genotype[i.age*2+1:i.age*2+2], e, ve))
    end

    function fitf(i::Individual, e)
        if i.age < nages
            i.fitness[1] = s[i.age+1]
        else
            i.fitness[1] = 0.0
        end
        i.fitness[2] = 1 + gauss_purify_cost_linear_norm(i.phenotype, e, [A, B], γ, γb, wmax)
    end

    function mutf(offspring::Individual, parent::Individual)
        copy!(offspring.genotype,
              parent.genotype + rand(Normal(0.0,vmut), size(parent.genotype)))
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
                   (e)->[1.0; autoregressive([e[2]],[arθ],stdenv)], # env update function
                   [1.0, 0.0]); # initial env state

    return p
end
