using PyPlot
using JLD

include("senescence.jl")

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

function eigSens(params)
    nages = length(params.f)
    # construct Leslie matrix
    projm = zeros(nages, nages)
    projm[1,:] = params.f
    projm += diagm(params.s, -1)

    # get right (w) and left (v) eigenvectors
    (e, w) = eig(projm)
    (e, v) = eig(projm')
    # find max absolute eigenvalue
    (absemax, imax) = findmax(abs.(e))
    println(e[imax])
    return v[:,imax] * w[:,imax]' / (w[:,imax]' * v[:,imax])
end


## timing
using BenchmarkTools

srand(3141);
pars = SenesceParams(n=500, s=[0.9, 0.9, 0.1, 0.1], ve=0.01,
                            A=1.0, B=1.0, γ=2.0, γb=0.1, wmax=3.0,
                            venv=0.1, arθ=0.75, vmut=0.01, reps=1, burns=1000, iters=1000)
p = popLinearPlasticityFert(pars);
@time (mg, env) = runSim(p, pars)

srand(3141);
pars = SenesceParams(n=500, s=[0.9, 0.9, 0.1, 0.1], ve=0.01,
                            A=1.0, B=1.0, γ=2.0, γb=0.1, wmax=3.0,
                            venv=0.1, arθ=0.75, vmut=0.01, reps=1, burns=1000, iters=1000)
p = popLinearPlasticityFert(pars);
@benchmark (mg, env) = runSim(p, pars)

## profiling

srand(3141);
pars = SenesceParams(n=500, s=[0.9, 0.9, 0.1, 0.0], ve=0.01,
                            A=1.0, B=1.0, γ=2.0, γb=0.1, wmax=3.0,
                            venv=0.1, arθ=0.75, vmut=0.01)
p = popLinearPlasticityFert(pars);
@profile (mg, env) = runSim(p, 1, 20000, 0);

srand(3141);
p = popLinearPlasticitySurv(n=500, s=[0.9, 0.9, 0.1, 0.0], ve=0.01,
                            A=1.0, B=1.0, γ=2.0, γb=0.1, wmax=3.0,
                            venv=0.1, arθ=0.75, vmut=0.01);
@profile (mg, env) = runSim(p, 1, 20000, 0);

### burn in length
run(`julia senescence_sim.jl --n=500 --s="[0.1, 0.4, 0.7, 0.9]" --f="[1.0, 1.0, 1.0, 1.0, 1.0]" --evolsf="[true,false]" --ve=0.01 --A=2.0 --B=2.0 --γ=0.1 --γb=0 --wmax=1.0 --venv=1 --arθ=0.75 --vmut=0.1 --reps=10 --burns=100 --iters=100 --file=test1.jld`);
run(`julia senescence_sim.jl --n=500 --s="[0.1, 0.4, 0.7, 0.9]" --f="[1.0, 1.0, 1.0, 1.0, 1.0]" --evolsf="[true,false]" --ve=0.01 --A=2.0 --B=2.0 --γ=0.1 --γb=0 --wmax=1.0 --venv=1 --arθ=0.75 --vmut=0.1 --reps=10 --burns=500 --iters=100 --file=test2.jld`);
run(`julia senescence_sim.jl --n=500 --s="[0.1, 0.4, 0.7, 0.9]" --f="[1.0, 1.0, 1.0, 1.0, 1.0]" --evolsf="[true,false]" --ve=0.01 --A=2.0 --B=2.0 --γ=0.1 --γb=0 --wmax=1.0 --venv=1 --arθ=0.75 --vmut=0.1 --reps=10 --burns=1000 --iters=100 --file=test3.jld`);
run(`julia senescence_sim.jl --n=500 --s="[0.1, 0.4, 0.7, 0.9]" --f="[1.0, 1.0, 1.0, 1.0, 1.0]" --evolsf="[true,false]" --ve=0.01 --A=2.0 --B=2.0 --γ=0.1 --γb=0 --wmax=1.0 --venv=1 --arθ=0.75 --vmut=0.1 --reps=10 --burns=5000 --iters=100 --file=test4.jld`);

nages = 4
clf()
(mg, env, params) = load("test1.jld", "mg", "env", "params");
subplot(421)
plotBoxPlot(mg[1:2:2*nages,:]', landeSlope(params.A, params.B, params.γ, params.γb, params.venv, params.arθ)[1], string.(range(0,10)), "black")
subplot(422)
plotBoxPlot(mg[2:2:2*nages,:]', landeSlope(params.A, params.B, params.γ, params.γb, params.venv, params.arθ)[2], string.(range(0,10)), "black")
(mg, env, params) = load("test2.jld", "mg", "env", "params");
subplot(423)
plotBoxPlot(mg[1:2:2*nages,:]', landeSlope(params.A, params.B, params.γ, params.γb, params.venv, params.arθ)[1], string.(range(0,10)), "black")
subplot(424)
plotBoxPlot(mg[2:2:2*nages,:]', landeSlope(params.A, params.B, params.γ, params.γb, params.venv, params.arθ)[2], string.(range(0,10)), "black")
(mg, env, params) = load("test3.jld", "mg", "env", "params");
subplot(425)
plotBoxPlot(mg[1:2:2*nages,:]', landeSlope(params.A, params.B, params.γ, params.γb, params.venv, params.arθ)[1], string.(range(0,10)), "black")
subplot(426)
plotBoxPlot(mg[2:2:2*nages,:]', landeSlope(params.A, params.B, params.γ, params.γb, params.venv, params.arθ)[2], string.(range(0,10)), "black")
(mg, env, params) = load("test4.jld", "mg", "env", "params");
subplot(427)
plotBoxPlot(mg[1:2:2*nages,:]', landeSlope(params.A, params.B, params.γ, params.γb, params.venv, params.arθ)[1], string.(range(0,10)), "black")
subplot(428)
plotBoxPlot(mg[2:2:2*nages,:]', landeSlope(params.A, params.B, params.γ, params.γb, params.venv, params.arθ)[2], string.(range(0,10)), "black")
