# Plasticity and senescence simulation
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

include("populations.jl")
include("environment.jl")
include("phenotype.jl")
include("fitness.jl")

using Distributions
using populations

###
### Code testing
###

##
## common routines
##

# round to nearest integer randomly
function rround(x)
    return floor(Int,x) + (rand() < x - floor(Int,x))
end

# set initial frequency
function setInitFreq(p, freq0, gvals = [1.0, 0.0])
    n = p.size
    k = rround(n * freq0)
    for i in 1:k
        p.members[i].genotype = [gvals[1]]
    end
    for i in k+1:n
        p.members[i].genotype = [gvals[2]]
    end
end

# measure fixation
function fixation(p, freq0, reps, iters, gvals = [1.0, 0.0])
    m = 0.0
    for r in 1:reps
        if r % round(reps / 10) == 0
            println(string(round(100*r/reps), "% iter ", r))
        end
        setInitFreq(p, freq0, gvals)
        for i in 1:iters
            next_gen(p)
        end
        m += mean_genotype(p) / reps
    end

    return m[1]
end

##
## fixation proabability test
##

function phenof(i,e)
    i.phenotype = [2.0]
end

function fitf(i,e)
    i.fitness[1] = 0.0
    i.fitness[2] = 1.0
end

function mutf(i)
    return copy(i.genotype)
end

p = Population(# population size
               100,
               # genotype->phenotype map
               phenof,
               # fitness function
               fitf,
               # mutation function
               mutf,
               # initial genotype function
               (i)->[0.0],
               (e)->[1.0], # env update function
               [1.0]); # initial env state

tic();
freq0 = 0.1:0.1:0.9
prob = Array(Float64, length(freq0))
for i=1:length(freq0)
    prob[i] = fixation(p, freq0[i], 200, 800)
end
toc();

# binomial distribution conf intervals
function binomialConf(fvec, n, alpha)
    len = length(fvec)
    conf = zeros((len,2))
    for i=1:len
        conf[i,1] = quantile(Binomial(n,fvec[i]), alpha/2) / n
        conf[i,2] = quantile(Binomial(n,fvec[i]), 1-alpha/2) / n
    end
    return conf
end

# plot with 95% confidence intervals
using PyPlot

conf = binomialConf(freq0, 200, 0.05)
plot(freq0, prob, color="black", linestyle="-")
plot(freq0, conf[1:end,1], color="black", linestyle="--")
plot(freq0, conf[1:end,2], color="black", linestyle="--")

##
## fixation probability: single advantageous allele
##

function phenof(i,e)
    i.phenotype = [0.0]
end

function fitf(i,e)
    i.fitness[1] = 0.0
    i.fitness[2] = 1.0 + i.genotype[1]
end

function mutf(i)
    return copy(i.genotype)
end

p = Population(# population size
               100,
               # genotype->phenotype map
               phenof,
               # fitness function
               fitf,
               # mutation function
               mutf,
               # initial genotype function
               (i)->[0.0],
               (e)->[1.0], # env update function
               [1.0]); # initial env state

tic();
freq0 = 0.1:0.1:0.9
prob = Array(Float64, length(freq0))
for i=1:length(freq0)
    prob[i] = fixation(p, freq0[i], 200, 800, [0.01, 0.0]) / 0.01
end
toc();


# plot with 95% confidence intervals
using PyPlot

function fixprob(x, n, s)
    return (1 - exp(- 2 * n * s * x)) / (1 - exp(- 2 * n * s))
end

conf = binomialConf(map((x)->fixprob(x, 100, 0.01), 0.1:0.1:0.9), 200, 0.05)
plot(freq0, prob, color="black", linestyle="-")
plot(freq0, conf[1:end,1], color="black", linestyle="--")
plot(freq0, conf[1:end,2], color="black", linestyle="--")


##
## No generation overlap and no age-specific phenotypes
##
function phenof(i,e)
    i.phenotype = [linear_norm(i.genotype, e, 0.01)]
end

function fitf(i,e)
    i.fitness[1] = 0.0
    i.fitness[2] = gauss_purify_linear_norm(i.phenotype, e, [2.0, 1.0], 3.0, 10.0)
end

function mutf(i)
    return i.genotype + rand(Normal(0.0,0.01), size(i.genotype))
end

p = Population(# population size
               500,
               # genotype->phenotype map
               phenof,
               # fitness function
               fitf,
               # mutation function
               mutf,
               # initial genotype function
               ()->(rand(2)-0.5)/2.0,
               (e)->[1.0; autoregressive([e[2]],[0.5],0.1)], # env update function
               [1.0, 0.0]); # initial env state

tic();
#@profile(
         println(mean_genotype(p))
         println(mean_phenotype(p))
         println("--")
         for i in 1:10000
         next_gen(p)
         if i % 1000 == 0
             println(p.mean_fit[2])
             println(mean_genotype(p))
             println(mean_phenotype(p))
             println(p.env_state)
             println("--")
         end
         end
#         )
toc();

#
# Generation overlap and age-specific reaction norm
# -- N.B. anonymous functions are slow in Julia right now (sucks), so write explicit functions for
# -- those that are calculated for every individual every generation.
#
function phenof(i,e)
    i.phenotype = [linear_norm(i.genotype[ min(2*i.age+1,19) : min(2*i.age+2,20) ], e, 0.01)]
end

# genotype->phenotype map: select genotype with age-specific expression (max age 9)
function fitf(i,e)
    i.fitness[1] = 0.75
    i.fitness[2] = gauss_purify_linear_norm(i.phenotype, e, [1.0, 2.5], 3.0, 10.0)
end

function mutf(i)
    return i.genotype + rand(Normal(0.0,0.01), size(i.genotype))
end

p = Population(# population size
               500,
               # genotype->phenotype map
               phenof,
               # fitness function
               fitf,
               # mutation function
               mutf,
               # initial genotype function
               ()->0.01*(rand(20)-0.5),
               # env update function
               (e)->[1.0; autoregressive([e[2]],[0.5],0.1)],
               # initial env state
               [1.0, 0.0]); 

tic();
#@profile(
         for i in 1:10000
         next_gen(p)
         # if i % 1000 == 0
         #     println(p.mean_fit[2])
         #     println(mean_genotype(p))
         #     println(mean_phenotype(p))
         #     println(p.env_state)
         #     println("--")
         # end
         end
#         )
toc();
