# test age-structured population simulation
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

include("populations.jl")
include("environment.jl")
include("phenotype.jl")
include("fitness.jl")

using Distributions
using populations

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
            println(string(round(100*r/reps), "% rep ", r))
        end
        setInitFreq(p, freq0, gvals)
        for i in 1:iters
            next_gen(p)
        end
        m += mean_genotype(p) / reps
    end

    return m[1]
end

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

##
## fixation proabability test
##

function phenof(i,e)
    i.phenotype[1] = 2.0
end

function fitf(i,e)
    i.fitness[1] = 0.0
    i.fitness[2] = 1.0
end

function mutf(offspring::Individual, parent::Individual)
    copy!(offspring.genotype, parent.genotype)
end

p = Population(# population size
               100,
               # genotype->phenotype map, number of phenotypes
               phenof, 1, 
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
    prob[i] = fixation(p, freq0[i], 500, 800)
end
toc();

# plot with 95% confidence intervals
using PyPlot

conf = binomialConf(freq0, 500, 0.05)
plot(freq0, freq0, color="blue", linestyle="-")
plot(freq0, prob, color="black", linestyle="-")
plot(freq0, conf[1:end,1], color="black", linestyle="--")
plot(freq0, conf[1:end,2], color="black", linestyle="--")

##
## fixation probability: single advantageous allele
##

function phenof(i,e)
    i.phenotype[1] = 0.0
end

function fitf(i,e)
    i.fitness[1] = 0.0
    i.fitness[2] = 1.0 + i.genotype[1]
end

function mutf(offspring::Individual, parent::Individual)
    copy!(offspring.genotype, parent.genotype)
end


p = Population(# population size
               100,
               # genotype->phenotype map, number of phenotypes
               phenof, 1, 
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
reps = 1000
prob = Array(Float64, length(freq0))
for i=1:length(freq0)
    prob[i] = fixation(p, freq0[i], reps, 800, [0.01, 0.0]) / 0.01
end
toc();


# plot with 95% confidence intervals using analytical fixation probability
using PyPlot

function fixprob(x, n, s)
    return (1 - exp(- 2 * n * s * x)) / (1 - exp(- 2 * n * s))
end

conf = binomialConf(map((x)->fixprob(x, 100, 0.01), 0.1:0.1:0.9), reps, 0.05)
plot(freq0, prob, color="blue", linestyle="-")
plot(freq0, map((x)->fixprob(x, 100, 0.01), 0.1:0.1:0.9), color="black", linestyle="-")
plot(freq0, conf[1:end,1], color="black", linestyle="--")
plot(freq0, conf[1:end,2], color="black", linestyle="--")


##
## fixtion probability of a neutral allele in an age-structured population
## using theory from Emigh (1979, Genetics 92:323--337) and Vindenes et al (2010, Evolution)
##

function phenof(i::Individual,e)
    @inbounds i.phenotype[1] = 0.0
end

s1 = 0.99
s2 = 0.8
s3 = 0.1

function fitf(i::Individual,e)
    # survival
    @inbounds begin
        if i.age == 0
            i.fitness[1] = s1
        elseif i.age == 1
            i.fitness[1] = s2
        elseif i.age == 2
            i.fitness[1] = s3
        else 
            i.fitness[1] = 0.0
        end
        # fertility 
        i.fitness[2] = 1.0
    end
end

function mutf(offspring::Individual, parent::Individual)
    copy!(offspring.genotype, parent.genotype)
end

function stableAgeDist(s1,s2,s3)
    n_d_n1 = 1 + s1 + s1*s2 + s1*s2*s3
    P = hcat([ 1 / n_d_n1, 1 / n_d_n1, 1 / n_d_n1, 1 / n_d_n1],
             [s1, 0, 0, 0],
             [0, s2, 0, 0],
             [0, 0, s3, 0])'
    e, v = eig(P)
    imax = indmax(abs(e))
    return v[:,imax] / sum(v[:,imax])
end

function fixationProbAge(s1,s2,s3,psize)
    n_d_n1 = 1 + s1 + s1*s2 + s1*s2*s3
    P = hcat([ 1 / n_d_n1, 1 / n_d_n1, 1 / n_d_n1, 1 / n_d_n1],
             [s1, 0, 0, 0],
             [0, s2, 0, 0],
             [0, 0, s3, 0])'
    re, rv = eig(P)
    le, lv = eig(P')
    rimax = indmax(abs(re))
    limax = indmax(abs(le))
    mrv = rv[:,rimax]
    mlv = lv[:,limax]
    
    return real(mlv / (mlv ⋅ mrv/sum(mrv)) / psize)
end

function setInitMut(p::Population, age, gvals = [1.0, 0.0])
    success = false
    for i = 1:p.size
        if p.members[i].age == age && !success
            p.members[i].genotype = [gvals[1]]
            success = true
        else
            p.members[i].genotype = [gvals[2]]
        end
    end

    return success
end

# measure fixation in neutral population
function fixation(p, age, reps, iters, gvals = [1.0, 0.0])
    tic()
    map((i)->next_gen(p), 1:1000); # iterate population to stable age dist for first run
    m = 0.0
    for r in 1:reps
        setInitMut(p, age, gvals)
        if r % round(reps / 10) == 0
            println(round(100*r/reps), "% rep ", r)
            toc()
            tic()
        end
        for i in 1:iters
            next_gen(p)
        end
        m += mean_genotype(p) / reps
    end
    toq()
    return m[1]
end

p = Population(# population size
               100,
               # genotype->phenotype map, number of phenotypes
               phenof, 1, 
               # fitness function
               fitf,
               # mutation function
               mutf,
               # initial genotype function
               (i)->[0.0],
               (e)->[1.0], # env update function
               [1.0]); # initial env state


ages = [0, 1, 2, 3]
reps = 10000
prob = Array(Float64, length(ages))
for i=1:length(ages)
    println("age: ", ages[i])
    prob[i] = fixation(p, ages[i], reps, 1000)
end

# plot with 95% confidence intervals
using PyPlot

eprob = fixationProbAge(s1,s2,s3,p.size)
conf = binomialConf(eprob, reps, 0.05)
plot(ages, eprob, color="blue", linestyle="-")
plot(ages, prob, color="black", linestyle="-")
plot(ages, conf[1:end,1], color="black", linestyle="--")
plot(ages, conf[1:end,2], color="black", linestyle="--")

##
## Continuum of alleles, no generation overlap, linear reaction norm with cost of slope.
## Test against theory from Lande (2014, JEB)
##

ve = 0.01
A = 1.0
B = 2.0
γ = 3.0
γb = 0.5
wmax = 3.0
venv = 0.1
arθ = 0.75
vmut = 0.01

function phenof(i::Individual,e)
    copy!(i.phenotype, linear_norm_g(i.genotype, e, ve))
    #i.phenotype = linear_norm_g(i.genotype, e, ve)
end

function fitf(i::Individual,e)
    i.fitness[1] = 0.0
    i.fitness[2] = gauss_purify_cost_linear_norm(i.phenotype, e, [A, B], γ, γb, wmax)
end

function mutf(offspring::Individual, parent::Individual)
    copy!(offspring.genotype, parent.genotype)
end

function mutf(offspring::Individual, parent::Individual)
    copy!(offspring.genotype,
          parent.genotype + rand(Normal(0.0,vmut), size(parent.genotype)))
end

p = Population(# population size
               250,
               # genotype->phenotype map
               phenof, 3,
               # fitness function
               fitf,
               # mutation function
               mutf,
               # initial genotype function
               (i)->zeros(2),
               (e)->[1.0; autoregressive([e[2]],[arθ],sqrt(venv))], # env update function
               [1.0, 0.0]); # initial env state

tic();
iters = 50000;
mg = zeros(size(mean_genotype(p)));
evec = zeros(iters);
for i in 1:iters
    next_gen(p)
    mg += mean_genotype(p) / iters
    evec[i] = p.env_state[2]
    if i % 1000 == 0
        println(p.mean_fit[2])
        println(mean_genotype(p))
        println(mean_phenotype(p))
        println(p.env_state)
        println("--")
    end
end
println(string("env var: ", var(evec)));
println(string("predicted env var: ", venv / (1 - arθ^2))); ## from variance of AR(1) process
println(string("mean genotype: ", mg));
println(string("predicted-actual mean genotype: ", [A, B / (1 + γb / (γ * venv / (1 - arθ^2)))] - mg[:])); ## from Lande (2014, JEB)
toc();

landeSlopes = (A,B,γ,γb,v) -> [A, B / (1 + γb / (γ * v))]
varAR = (arθ, venv) -> venv / (1 - arθ^2)
