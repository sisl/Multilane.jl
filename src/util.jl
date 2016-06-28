#NOTE: from StatsBase
function sample(rng::AbstractRNG,wv::WeightVec)
    t = rand(rng) * sum(wv)
    w = values(wv)
    n = length(w)
    i = 1
    cw = w[1]
    while cw < t && i < n
            i += 1
            @inbounds cw += w[i]
    end
    return i
end
sample(rng::AbstractRNG, a::AbstractArray, wv::WeightVec) = a[sample(rng,wv)]
