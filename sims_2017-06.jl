using PyPlot
using JLD

include("senescence.jl")

reps = 50
burns = 10000
iters = 1000

γb_vec = [0., 0.5, 1.0, 1.5];
pops = Array(Population, length(γb_vec))
mgs = Array(Any, length(γb_vec))

for i = 1:length(γb_vec)
    println("iter ", i)
    pops[i] = popLinearPlasticityFert(n=500, s=[0.9, 0.9, 0.1, 0.0], ve=0.01,
                                      A=1.0, B=1.0, γ=2.0, γb=γb_vec[i], wmax=3.0,
                                      venv=0.1, arθ=0.75, vmut=0.01);
    @time (mg, env) = runSim(pops[i], reps, burns, iters);
    mgs[i] = mg
end

ls = transpose(hcat(map((x)->landeSlope(1., 1., 2., x, 0.1, 0.75), γb_vec)...))

## Plot intercept for the different cost values
clf()
plot(ls[:,1])
plot(hcat(map((x)->mean(x[1:2:7,:],2), mgs)...))

clf()
subplot(221)
plotBoxPlot(mgs[1][1:2:7,:]', landeSlope(1., 1., 2., γb_vec[1], 0.1, 0.75)[1], ["0", "1", "2", "3"], "black")
subplot(222)
plotBoxPlot(mgs[2][1:2:7,:]', landeSlope(1., 1., 2., γb_vec[2], 0.1, 0.75)[1], ["0", "1", "2", "3"], "black")
subplot(223)
plotBoxPlot(mgs[3][1:2:7,:]', landeSlope(1., 1., 2., γb_vec[3], 0.1, 0.75)[1], ["0", "1", "2", "3"], "black")
subplot(224)
plotBoxPlot(mgs[4][1:2:7,:]', landeSlope(1., 1., 2., γb_vec[4], 0.1, 0.75)[1], ["0", "1", "2", "3"], "black")

## Plot slope for the different cost values
clf()
plot(ls[:,2])
plot(hcat(map((x)->mean(x[2:2:8,:],2), mgs)...))

clf()
subplot(221)
plotBoxPlot(mgs[1][2:2:8,:]', landeSlope(1., 1., 2., γb_vec[1], 0.1, 0.75)[2], ["0", "1", "2", "3"], "black")
subplot(222)
plotBoxPlot(mgs[2][2:2:8,:]', landeSlope(1., 1., 2., γb_vec[2], 0.1, 0.75)[2], ["0", "1", "2", "3"], "black")
subplot(223)
plotBoxPlot(mgs[3][2:2:8,:]', landeSlope(1., 1., 2., γb_vec[3], 0.1, 0.75)[2], ["0", "1", "2", "3"], "black")
subplot(224)
plotBoxPlot(mgs[4][2:2:8,:]', landeSlope(1., 1., 2., γb_vec[4], 0.1, 0.75)[2], ["0", "1", "2", "3"], "black")




save("fert_no_cost_01.jld", "mg", mg, "env", env)

mg, env = load("fert_no_cost_01.jld", "mg", "env");
clf()
plotBoxPlot(mg[1:2:7,:]', landeSlope(A, B, γ, γb, venv, arθ)[1], ["0", "1", "2", "3"], "black")
savefig("fert_no_cost_intercept_01.pdf", bbox_inches="tight")
clf()
plotBoxPlot(mg[2:2:8,:]', landeSlope(A, B, γ, γb, venv, arθ)[2], ["0", "1", "2", "3"], "black")
savefig("fert_no_cost_slope_01.pdf", bbox_inches="tight")


## timing
using BenchmarkTools

srand(3141);
println("v1")
p = popLinearPlasticityFert(n=500, s=[0.9, 0.9, 0.1, 0.0], ve=0.01,
                            A=1.0, B=1.0, γ=2.0, γb=0.1, wmax=3.0,
                            venv=0.1, arθ=0.75, vmut=0.01);
@benchmark (mg, env) = runSim(p, 1, 1000, 0, false)

## profiling

srand(3141);
p = popLinearPlasticityFert(n=500, s=[0.9, 0.9, 0.1, 0.0], ve=0.01,
                            A=1.0, B=1.0, γ=2.0, γb=0.1, wmax=3.0,
                            venv=0.1, arθ=0.75, vmut=0.01);
@profile (mg, env) = runSim(p, 1, 20000, 0);

srand(3141);
p = popLinearPlasticitySurv(n=500, s=[0.9, 0.9, 0.1, 0.0], ve=0.01,
                            A=1.0, B=1.0, γ=2.0, γb=0.1, wmax=3.0,
                            venv=0.1, arθ=0.75, vmut=0.01);
@profile (mg, env) = runSim(p, 1, 20000, 0);
