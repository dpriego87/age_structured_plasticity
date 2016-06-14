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

## ve = 0.01
## A = 1.0
## B = 2.0
## γ = 3.0
## γb = 0.5
## wmax = 3.0
## venv = 0.1
## arθ = 0.75
## vmut = 0.01

function linearPlasticityFertAgePop(n, s, ve, A, B, γ, γb, wmax, venv, arθ, vmut)
    nages = length(s)

    function phenof(i::Individual, e)
        copy!(i.phenotype, linear_norm_g(i.genotype[i.age+1:i.age+2], e, ve))
    end

    function fitf(i::Individual, e)
        if i.age < nages
            i.fitness[1] = s[i.age+1]
        else
            i.fitness[1] = 0.0
        end
        i.fitness[2] = gauss_purify_cost_linear_norm(i.phenotype, e, [A, B], γ, γb, wmax)
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
                   (e)->[1.0; autoregressive([e[2]],[arθ],sqrt(venv))], # env update function
                   [1.0, 0.0]); # initial env state

    return p
end

function landeSlope(A, B, γ, γb, venv, arθ)
    # variance for autoregressive process of order one
    v = venv / (1 - arθ^2)
    # predicted slopes from Lande (2014, JEB)
    return [A, B / (1 + γb / (γ * v))]
end

function runSim(p, reps, burns, iters)
    ngeno = length(p.members[1].genotype)
    nenv  = length(p.env_state)
    m1geno = zeros((ngeno, iters))
    m2geno = zeros((ngeno, iters))
    m1env  = zeros((nenv, iters))
    m2env  = zeros((nenv, iters))

    for r = 1:reps
        # reset genotypes
        for i in 1:p.size
            p.members[i].genotype = zeros(ngeno)
        end
        # burn-in
        for j = 1:burns
            next_gen(p)
        end
        # main iteration
        for j = 1:iters
            next_gen(p)
            m1geno[:,r] = mean_genotype(p) / iters
            m2geno[:,r] = mean_genotype(p).^2 / iters
            m1env[:,r]  = p.env_state / iters
            m2env[:,r]  = p.env_state.^2 / iters
        end
        if r % round(reps / 100) == 0
            print(round(Int,100*r/reps), "% ")
        end
    end

    return (m1geno, m2geno-m1geno.^2, m1geno, m2env-m1env.^2)
end

