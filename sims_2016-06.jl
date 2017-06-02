using PyPlot
using JLD

## Preliminary runs with fertility effects -- run 01

n = 500
s = [0.9, 0.9, 0.1, 0.0]
ve = 0.01
A = 2.0
B = 2.0
γ = 3.0
γb = 0.0
wmax = 1.0
venv = 0.1
arθ = 0.9
vmut = 0.01

reps = 100
burns = 20000
iters = 1000

# no cost
p = linearPlasticityFertAgePop(n, s, ve, A, B, γ, γb, wmax, venv, arθ, vmut);
@time (mg, env) = runSim(p, reps, burns, iters);
save("fert_no_cost_01.jld", "mg", mg, "env", env)

mg, env = load("fert_no_cost_01.jld", "mg", "env");
clf()
plotBoxPlot(mg[1:2:7,:]', landeSlope(A, B, γ, γb, venv, arθ)[1], ["0", "1", "2", "3"], "black")
savefig("fert_no_cost_intercept_01.pdf", bbox_inches="tight")
clf()
plotBoxPlot(mg[2:2:8,:]', landeSlope(A, B, γ, γb, venv, arθ)[2], ["0", "1", "2", "3"], "black")
savefig("fert_no_cost_slope_01.pdf", bbox_inches="tight")

# cost
γb = 1.0
p = linearPlasticityFertAgePop(n, s, ve, A, B, γ, γb, wmax, venv, arθ, vmut);
@time (mg, env) = runSim(p, reps, burns, iters);
save("fert_cost_01.jld", "mg", mg, "env", env)

mg, env = load("fert_cost_01.jld", "mg", "env");
clf()
plotBoxPlot(mg[1:2:7,:]', landeSlope(A, B, γ, γb, venv, arθ)[1], ["0", "1", "2", "3"], "black")
savefig("fert_cost_intercept_01.pdf", bbox_inches="tight")
clf()
plotBoxPlot(mg[2:2:8,:]', landeSlope(A, B, γ, γb, venv, arθ)[2], ["0", "1", "2", "3"], "black")
savefig("fert_cost_slope_01.pdf", bbox_inches="tight")

## Preliminary runs with survival effects -- run 01

n = 500
s = [0.9, 0.9, 0.1, 0.0]
ve = 0.01
A = 2.0
B = 2.0
γ = 3.0
γb = 0.0
wmax = 0.1
venv = 0.1
arθ = 0.9
vmut = 0.01

reps = 100
burns = 20000
iters = 1000

# no cost
p = linearPlasticitySurvAgePop(n, s, ve, A, B, γ, γb, wmax, venv, arθ, vmut);
@time (mg, env) = runSim(p, reps, burns, iters);
save("surv_no_cost_01.jld", "mg", mg, "env", env)

mg, env = load("surv_no_cost_01.jld", "mg", "env");
clf()
plotBoxPlot(mg[1:2:6,:]', landeSlope(A, B, γ, γb, venv, arθ)[1], ["0", "1", "2", "3"], "black")
savefig("surv_no_cost_intercept_01.pdf", bbox_inches="tight")
clf()
plotBoxPlot(mg[2:2:6,:]', landeSlope(A, B, γ, γb, venv, arθ)[2], ["0", "1", "2", "3"], "black")
savefig("surv_no_cost_slope_01.pdf", bbox_inches="tight")

# cost
γb = 1.0
p = linearPlasticityFertAgePop(n, s, ve, A, B, γ, γb, wmax, venv, arθ, vmut);
@time (mg, env) = runSim(p, reps, burns, iters);
save("surv_cost_01.jld", "mg", mg, "env", env)

mg, env = load("surv_cost_01.jld", "mg", "env");
clf()
plotBoxPlot(mg[1:2:6,:]', landeSlope(A, B, γ, γb, venv, arθ)[1], ["0", "1", "2", "3"], "black")
savefig("surv_cost_intercept_01.pdf", bbox_inches="tight")
clf()
plotBoxPlot(mg[2:2:6,:]', landeSlope(A, B, γ, γb, venv, arθ)[2], ["0", "1", "2", "3"], "black")
savefig("surv_cost_slope_01.pdf", bbox_inches="tight")
