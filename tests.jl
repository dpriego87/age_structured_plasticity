# test age-structured population simulation
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

include("populations.jl")
include("environment.jl")
include("phenotype.jl")
include("fitness.jl")

using Distributions, LinearAlgebra
using PyPlot
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
    m = zeros(length(p.members[1].genotype))
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

# tic();
freq0 = 0.1:0.1:0.9
prob = Array{Float64}(undef, length(freq0))
@time for i=1:length(freq0)
    prob[i] = fixation(p, freq0[i], 500, 800)
end
# toc();

# plot with 95% confidence intervals
conf = binomialConf(freq0, 500, 0.05)
pygui(true)
plot(freq0, freq0, color="blue", linestyle="-")
plot(freq0, prob, color="black", linestyle="-")
plot(freq0, conf[1:end,1], color="black", linestyle="--")
plot(freq0, conf[1:end,2], color="black", linestyle="--")
gcf()
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

# tic();
freq0 = 0.1:0.1:0.9
reps = 1000
prob = Array{Float64}(undef, length(freq0))
@time for i=1:length(freq0)
    prob[i] = fixation(p, freq0[i], reps, 800, [0.01, 0.0]) / 0.01
end
# toc();


# plot with 95% confidence intervals using analytical fixation probability
function fixprob(x, n, s)
    return (1 - exp(- 2 * n * s * x)) / (1 - exp(- 2 * n * s))
end

conf = binomialConf(map((x)->fixprob(x, 100, 0.01), 0.1:0.1:0.9), reps, 0.05)
pygui(true)
plot(freq0, prob, color="blue", linestyle="-")
plot(freq0, map((x)->fixprob(x, 100, 0.01), 0.1:0.1:0.9), color="black", linestyle="-")
plot(freq0, conf[1:end,1], color="black", linestyle="--")
plot(freq0, conf[1:end,2], color="black", linestyle="--")
gcf()

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
    re, rv = eigen(P)
    le, lv = eigen(P')
    rimax = findmax(abs.(re))[2]
    limax = findmax(abs.(le))[2]
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
    # tic()
    map((i)->next_gen(p), 1:1000); # iterate population to stable age dist for first run
    m = zeros(length(p.members[1].genotype))
    for r in 1:reps
        setInitMut(p, age, gvals)
        if r % round(reps / 10) == 0
            println(round(100*r/reps), "% rep ", r)
            # toc()
            # tic()
        end
        for i in 1:iters
            next_gen(p)
        end
        m += mean_genotype(p) / reps
    end
    # toq()
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
prob = Array{Float64}(undef,length(ages))
for i=1:length(ages)
    println("age: ", ages[i])
    prob[i] = @time fixation(p, ages[i], reps, 1000)
end

# plot with 95% confidence intervals
eprob = fixationProbAge(s1,s2,s3,p.size)
conf = binomialConf(eprob, reps, 0.05)
pygui(true)
plot(ages, eprob, color="blue", linestyle="-")
plot(ages, prob, color="black", linestyle="-")
plot(ages, conf[1:end,1], color="black", linestyle="--")
plot(ages, conf[1:end,2], color="black", linestyle="--")
gcf()
##
## Continuum of alleles, no generation overlap, linear reaction norm with cost of slope.
## Test against theory from Lande (2014, JEB)
##

function linearPlasticity(n, ve, A, B, γ, γb, wmax, venv, arθ, vmut)
    stdenv = sqrt(venv)

    function phenof(i::Individual, e)
        copy!(i.phenotype, linear_norm_g(i.genotype, e, ve))
    end

    function fitf(i::Individual, e)
        i.fitness[1] = 0.0
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
                   (i)->zeros(2),
                   (e)->[1.0; autoregressive([e[2]],[arθ],stdenv)], # env update function
                   [1.0, 0.0]); # initial env state

    return p
end


function runSim(p, reps, burns, iters)
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
            if ((r-1)*totalruns+j) % iterblock == 0
                print(round(Int,100.0*((r-1)*totalruns+j)/(totalruns*reps)), "% ")
            end
        end
    end

    return (mgeno, env)
end

res = zeros(4,4)
p = linearPlasticity(500, 0.01, 1., 2., 3., 0.0, 3., 0.1, 0.75, 0.01);
@time (mg, env) = runSim(p, 10, 10000, 1000);
res[1,1:2] = mean(mg,2);
res[1,3:4] = landeSlope(1., 2., 3., 0.0, 0.1, 0.75);

p = linearPlasticity(500, 0.01, 1., 2., 3., 0.5, 3., 0.1, 0.75, 0.01);
@time (mg, env) = runSim(p, 10, 10000, 1000);
res[2,1:2] = mean(mg,2);
res[2,3:4] = landeSlope(1., 2., 3., 0.5, 0.1, 0.75);

p = linearPlasticity(500, 0.01, 1., 2., 3., 1.0, 3., 0.1, 0.75, 0.01);
@time (mg, env) = runSim(p, 10, 10000, 1000);
res[3,1:2] = mean(mg,2);
res[3,3:4] = landeSlope(1., 2., 3., 1.0, 0.1, 0.75);

p = linearPlasticity(500, 0.01, 1., 2., 3., 2.0, 3., 0.1, 0.75, 0.01);
@time (mg, env) = runSim(p, 10, 10000, 1000);
res[4,1:2] = mean(mg,2);
res[4,3:4] = landeSlope(1., 2., 3., 2.0, 0.1, 0.75);

plot(res)
