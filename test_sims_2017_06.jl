using PyPlot
using JLD

include("senescence.jl")

reps = 50
burns = 10000
iters = 1000

γb_vec = [0., 0.5, 1.0, 1.5];
pops = Array(Population, length(γb_vec))
mgs = Array(Any, length(γb_vec))

srand(3141);
for i = 1:length(γb_vec)
    println("iter ", i)
    pops[i] = popLinearPlasticityFert(n=500, s=[0.9, 0.9, 0.1, 0.0], ve=0.01,
                                      A=1.0, B=1.0, γ=2.0, γb=γb_vec[i], wmax=3.0,
                                      venv=0.1, arθ=0.75, vmut=0.01);
    # pops[i] = linearPlasticityFertAgePop(500, [0.9, 0.9, 0.1, 0.0], 0.01,
    #                                      1.0, 1.0, 2.0, γb_vec[i], 3.0,
    #                                      0.1, 0.75, 0.01);
    @time (mg, env) = runSim(pops[i], reps, burns, iters);
    mgs[i] = mg
end

ls = transpose(hcat(map((x)->landeSlope(1., 1., 2., x, 0.1, 0.75), γb_vec)...))

## Plot intercept for the different cost values
figure()
subplot(221)
plotBoxPlot(mgs[1][1:2:7,:]', landeSlope(1., 1., 2., γb_vec[1], 0.1, 0.75)[1], ["0", "1", "2", "3"], "black")
subplot(222)
plotBoxPlot(mgs[2][1:2:7,:]', landeSlope(1., 1., 2., γb_vec[2], 0.1, 0.75)[1], ["0", "1", "2", "3"], "black")
subplot(223)
plotBoxPlot(mgs[3][1:2:7,:]', landeSlope(1., 1., 2., γb_vec[3], 0.1, 0.75)[1], ["0", "1", "2", "3"], "black")
subplot(224)
plotBoxPlot(mgs[4][1:2:7,:]', landeSlope(1., 1., 2., γb_vec[4], 0.1, 0.75)[1], ["0", "1", "2", "3"], "black")

## Plot slope for the different cost values
figure()
clf()
subplot(221)
plotBoxPlot(mgs[1][2:2:8,:]', landeSlope(1., 1., 2., γb_vec[1], 0.1, 0.75)[2], ["0", "1", "2", "3"], "black")
subplot(222)
plotBoxPlot(mgs[2][2:2:8,:]', landeSlope(1., 1., 2., γb_vec[2], 0.1, 0.75)[2], ["0", "1", "2", "3"], "black")
subplot(223)
plotBoxPlot(mgs[3][2:2:8,:]', landeSlope(1., 1., 2., γb_vec[3], 0.1, 0.75)[2], ["0", "1", "2", "3"], "black")
subplot(224)
plotBoxPlot(mgs[4][2:2:8,:]', landeSlope(1., 1., 2., γb_vec[4], 0.1, 0.75)[2], ["0", "1", "2", "3"], "black")


savefig("testnewA.pdf", bbox_inches="tight")
savefig("testnewB.pdf", bbox_inches="tight")

save("fert_no_cost_01.jld", "mg", mg, "env", env)
mg, env = load("fert_no_cost_01.jld", "mg", "env");


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
