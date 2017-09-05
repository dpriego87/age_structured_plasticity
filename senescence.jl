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

## Parameter object
type SenesceParams
    n::Int64
    s::Array{Float64,1}
    f::Array{Float64,1}
    evolsf::Array{Bool,1}
    ve::Float64
    A::Float64
    B::Float64
    γ::Float64
    γb::Float64
    wmax::Float64
    venv::Float64
    arθ::Float64
    vmut::Float64
    reps::Int64
    burns::Int64
    iters::Int64

    # Constructor. Takes keyword arguments to make it easier to call with command line arguments
    SenesceParams(;n::Int64=100, s::Array{Float64,1}=[0.0], f::Array{Float64,1}=[1.0, 1.0],
                  evolsf::Array{Bool,1}=[false,true],
                  ve::Float64=0.01,
                  A::Float64=1.0, B::Float64=1.0, γ::Float64=2.0, γb::Float64=0.0,
                  wmax::Float64=1.0, venv::Float64=0.1, arθ::Float64=0.75, vmut::Float64=0.01,
                  reps::Int64=10, burns::Int64=5000, iters::Int64=1000) =
                      length(f) != length(s) + 1 &&
                      (evolsf[1] || evolsf[2]) && !(evolsf[1] && evolsf[2]) ?
                      error("invalid parameter construction") :
                      new(n, s, f, evolsf, ve, A, B, γ, γb, wmax, venv, arθ, vmut, reps, burns, iters)
end

function popLinearPlasticity(params)
    # nages is number of stages with positive survival, i.e.: w/o terminal age class
    nages = length(params.s)
    # number of genes, which is either the twice number of surivival or fertility parameters
    # (only one element of params.evolsf should be true)
    ngenes = 2 * (params.evolsf[1] * length(params.s) + params.evolsf[2] * length(params.f))
    stdenv = sqrt(params.venv)
    stde = sqrt(params.ve)
    smut = sqrt(params.vmut)
    AB = [params.A, params.B]
    arθ = [params.arθ]
    
    function phenof(i::Individual, e::Array{Float64,1})
        if 2 * i.age < ngenes
            @views copy!(i.phenotype, linear_norm(i.genotype[i.age*2+1:i.age*2+2], e, stde))
        end
    end

    function fitf(i::Individual, e::Array{Float64,1})
        if i.age < nages
            i.fitness[1] = params.s[i.age+1]
            if params.evolsf[1]
                # evolve baseline survival
                i.fitness[1] +=
                    @views gauss_purify_cost_linear_norm(i.genotype[i.age*2+1:i.age*2+2],
                                                         i.phenotype,
                                                         e, AB, params.γ, params.γb, params.wmax)
            end
        else
            # terminal age class always dies
            i.fitness[1] = 0.0
        end
        # all ages including terminal age class can have positive fertility
        i.fitness[2] = params.f[i.age+1]
        if params.evolsf[2]
            # evolve baseline fertility
            i.fitness[2] +=
                @views gauss_purify_cost_linear_norm(i.genotype[i.age*2+1:i.age*2+2],
                                                     i.phenotype,
                                                     e, AB, params.γ, params.γb, params.wmax)
        end
    end

    function mutf(offspring::Individual, parent::Individual)
        copy!(offspring.genotype,
              parent.genotype + rand(Normal(0.0,smut), size(parent.genotype)))
    end

    function envf(e::Array{Float64,1})
        env = zeros(2)
        env[1] = 1.0
        env[2] = autoregressive([e[2]], arθ, stdenv)[1]
        return env
    end

    p = Population(# population size
                   params.n,
                   # genotype->phenotype map
                   phenof, 1,
                   # fitness function
                   fitf,
                   # mutation function
                   mutf,
                   # initial genotype function: fertility effects includes terminal age class
                   (i)->zeros(ngenes),
                   envf, # env update function
                   [1.0, 0.0]); # initial env state

    return p
end

function runSim(p, params, verbose=true)
    reps, burns, iters = (params.reps, params.burns, params.iters)
    
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
    println()

    return (mgeno, env)
end
