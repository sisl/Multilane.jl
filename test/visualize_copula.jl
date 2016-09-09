using Multilane
using Plots
using Base.Test

ndim = 3
N = 10000
rng = MersenneTwister(123)
c = GaussianCopula(ndim, 0.999)
@show c.cov

samples = Array(Float64, N, ndim)
@time for i in 1:N
    samples[i,:] = rand(rng, c)
end

@test all(samples.<=1.0)
@test all(samples.>=0.0)

#= 

plts = Array(Any, ndim, ndim)
for i in 1:ndim
    for j in 1:ndim
        plts[i,j] = scatter(samples[:,i], samples[:,j], label="($i,$j)", xlim=(0,1), ylim=(0,1))
    end
end
ss = plot(vec(plts)..., layout=(ndim, ndim))
gui(ss)

=#

hists = Array(Any, ndim)
for i in 1:ndim
    hists[i] = histogram(samples[:,i], label="$i")
end
hs = plot(hists..., layout = (ndim,1))
gui(hs)
