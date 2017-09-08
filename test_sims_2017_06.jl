using PyPlot
using JLD

include("senescence.jl")

function plotBoxPlot(vals, expect, xticklab, ycolor)
    gcf()[:set_size_inches](4,3)
    ax = gca()
    bp = PyPlot.boxplot(vals, sym="")
    setp(bp["boxes"], color="black")
    setp(bp["medians"], color=ycolor)
    setp(bp["whiskers"], color="black", dashes=(3,3))
    
    PyPlot.plot(0:size(vals)[2]+1, fill(expect,size(vals)[2]+2), linestyle="--", dashes=(5,5), color=ycolor)
    
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

function eigSens(s, f)
    nages = length(f)
    # construct Leslie matrix
    projm = zeros(nages, nages)
    projm[1,:] = f
    projm += diagm(s, -1)

    # get right (w) and left (v) eigenvectors
    (e, w) = eig(projm)
    (e, v) = eig(projm')
    # find max absolute eigenvalue
    (absemax, imax) = findmax(abs.(e))
    println("abs max eig: ", e[imax])
    
    return real(v[:,imax] * w[:,imax]' / (w[:,imax]' * v[:,imax]))
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

#### survival and fertility

run(`julia senescence_sim.jl --n=1000 --s="[0.9, 0.7, 0.5, 0.3, 0.1]" --f="[0.01, 0.01, 1.0, 1.0, 1.0, 1.0]" --evolsf="[true,false]" --ve=0.1 --A=2.0 --B=2.0 --γ=0.5 --γb=0 --wmax=0.075 --venv=1.0 --arθ=0.75 --vmut=0.0025 --reps=100 --burns=3000 --iters=100 --file=test_survI.jld`)
run(`julia senescence_sim.jl --n=1000 --s="[0.1, 0.3, 0.5, 0.7, 0.9]" --f="[0.01, 0.01, 1.0, 1.0, 1.0, 1.0]" --evolsf="[true,false]" --ve=0.1 --A=2.0 --B=2.0 --γ=0.5 --γb=0 --wmax=0.075 --venv=1.0 --arθ=0.75 --vmut=0.0025 --reps=100 --burns=3000 --iters=100 --file=test_survII.jld`)
run(`julia senescence_sim.jl --n=1000 --s="[0.9, 0.7, 0.5, 0.3, 0.1]" --f="[0.01, 0.01, 1.0, 1.0, 1.0, 1.0]" --evolsf="[false,true]" --ve=0.1 --A=2.0 --B=2.0 --γ=0.5 --γb=0 --wmax=0.075 --venv=1.0 --arθ=0.75 --vmut=0.0025 --reps=100 --burns=3000 --iters=100 --file=test_fertI.jld`)
run(`julia senescence_sim.jl --n=1000 --s="[0.1, 0.3, 0.5, 0.7, 0.9]" --f="[0.01, 0.01, 1.0, 1.0, 1.0, 1.0]" --evolsf="[false,true]" --ve=0.1 --A=2.0 --B=2.0 --γ=0.5 --γb=0 --wmax=0.075 --venv=1.0 --arθ=0.75 --vmut=0.0025 --reps=100 --burns=3000 --iters=100 --file=test_fertII.jld`)

(mg_s1, env_s1, params_s1) = load("test_survI.jld", "mg", "env", "params");
(mg_s2, env_s2, params_s2) = load("test_survII.jld", "mg", "env", "params");
(mg_f1, env_f1, params_f1) = load("test_fertI.jld", "mg", "env", "params");
(mg_f2, env_f2, params_f2) = load("test_fertII.jld", "mg", "env", "params");

ngenes_s1 = 2 * (params_s1.evolsf[1] * length(params_s1.s) + params_s1.evolsf[2] * length(params_s1.f))
ngenes_s2 = 2 * (params_s2.evolsf[1] * length(params_s2.s) + params_s2.evolsf[2] * length(params_s2.f))
ngenes_f1 = 2 * (params_f1.evolsf[1] * length(params_f1.s) + params_f1.evolsf[2] * length(params_f1.f))
ngenes_f2 = 2 * (params_f2.evolsf[1] * length(params_f2.s) + params_f2.evolsf[2] * length(params_f2.f))

subplot(221); plotBoxPlot(mg_s1[2:2:ngenes_s1,:]', landeSlope(params_s1.A, params_s1.B, params_s1.γ, params_s1.γb, params_s1.venv, params_s1.arθ)[2], string.(range(0,10)), "black")
subplot(222); plotBoxPlot(mg_s2[2:2:ngenes_s2,:]', landeSlope(params_s2.A, params_s2.B, params_s2.γ, params_s2.γb, params_s2.venv, params_s2.arθ)[2], string.(range(0,10)), "black")
subplot(223); plotBoxPlot(mg_f1[2:2:ngenes_f1,:]', landeSlope(params_f1.A, params_f1.B, params_f1.γ, params_f1.γb, params_f1.venv, params_f1.arθ)[2], string.(range(0,10)), "black")
subplot(224); plotBoxPlot(mg_f2[2:2:ngenes_f2,:]', landeSlope(params_f2.A, params_f2.B, params_f2.γ, params_f2.γb, params_f2.venv, params_f2.arθ)[2], string.(range(0,10)), "black")

### Plot strength of selection from matrix sensitivies

clf()
plot(((x)->x/sum(x))(eigSens(params_s1.s, params_s1.f)[1,:]), "k--")
plot(((x)->x/sum(x))(eigSens(params_s2.s, params_s2.f)[1,:]), "b--")
plot(((x)->x/sum(x))(diag(eigSens(params_s1.s, params_s1.f), -1)), "k-")
plot(((x)->x/sum(x))(diag(eigSens(params_s2.s, params_s2.f), -1)), "b-")


### use Seaborn to do boxplots

using Seaborn
using Pandas

(mg, env, params) = load("test_survI.jld", "mg", "env", "params");
ngenes = 2 * (params.evolsf[1] * length(params.s) + params.evolsf[2] * length(params.f))

df1 = melt(DataFrame(mg[1:2:ngenes,:]', columns=range(0,convert(Int,ngenes/2))),
           value_vars=columns(df), var_name="age", value_name="genotypic_value")
df1["param"] = "intercept"
df2 = melt(DataFrame(mg[2:2:ngenes,:]', columns=range(0,convert(Int,ngenes/2))),
           value_vars=columns(df), var_name="age", value_name="genotypic_value")
df2["param"] = "slope"
Seaborn.boxplot(x="age", y="genotypic_value", hue="param", data=concat((df1,df2)))
