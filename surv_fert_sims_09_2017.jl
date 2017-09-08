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

#### run survival and fertility simulations

survI   = "[0.9, 0.7, 0.5, 0.3, 0.1]"
survII  = "[0.1, 0.3, 0.5, 0.7, 0.9]"
fertI   = "[0.01, 0.01, 1.0, 1.0, 1.0, 1.0]"
fertII  = "[0.01, 0.01, 1.0, 1.0, 1.0, 1.0]"

date = string(Dates.Date(now()))
run(`julia senescence_sim.jl --n=1000 --s="$(survI)" --f="$(fertI)" --evolsf="[true,false]" --ve=0.1 --A=2.0 --B=2.0 --γ=0.5 --γb=0 --wmax=0.075 --venv=1.0 --arθ=0.75 --vmut=0.0025 --reps=100 --burns=3000 --iters=100 --file=sim_$(date)_survI.jld`)
run(`julia senescence_sim.jl --n=1000 --s="$(survII)" --f="$(fertII)" --evolsf="[true,false]" --ve=0.1 --A=2.0 --B=2.0 --γ=0.5 --γb=0 --wmax=0.075 --venv=1.0 --arθ=0.75 --vmut=0.0025 --reps=100 --burns=3000 --iters=100 --file=sim_$(date)_survII.jld`)
run(`julia senescence_sim.jl --n=1000 --s="$(survI)" --f="$(fertI)" --evolsf="[false,true]" --ve=0.1 --A=2.0 --B=2.0 --γ=0.5 --γb=0 --wmax=0.075 --venv=1.0 --arθ=0.75 --vmut=0.0025 --reps=100 --burns=3000 --iters=100 --file=sim_$(date)_fertI.jld`)
run(`julia senescence_sim.jl --n=1000 --s="$(survII)" --f="$(fertII)" --evolsf="[false,true]" --ve=0.1 --A=2.0 --B=2.0 --γ=0.5 --γb=0 --wmax=0.075 --venv=1.0 --arθ=0.75 --vmut=0.0025 --reps=100 --burns=3000 --iters=100 --file=sim_$(date)_fertII.jld`)

(mg_s1, env_s1, params_s1) = load("sim_$(date)_survI.jld", "mg", "env", "params");
(mg_s2, env_s2, params_s2) = load("sim_$(date)_survII.jld", "mg", "env", "params");
(mg_f1, env_f1, params_f1) = load("sim_$(date)_fertI.jld", "mg", "env", "params");
(mg_f2, env_f2, params_f2) = load("sim_$(date)_fertII.jld", "mg", "env", "params");

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

