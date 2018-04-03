"""
Use bisection search to sample efficiently given a vector of objects and the cumulative weights.
"""
function sample_cweighted(rng::AbstractRNG, objects::AbstractVector, cweights::AbstractVector{Float64})
    t = rand(rng)*cweights[end]
    large = length(cweights) # index of cdf value that is bigger than t
    small = 0 # index of cdf value that is smaller than t
    while large > small + 1
        new = div(small + large, 2)
        if t < cweights[new]
            large = new
        else
            small = new
        end
    end
    return objects[large]
end

# XXX Not complete
# """
# Perform low variance resampling from page 110 of Probabilistic Robotics by Thrun, Burgard, and Fox
# """
# function low_var_resample(rng, b::BehaviorParticleBelief, n)
#     for 
#     ps = Array{S}(n)
#     r = rand(rng)*last(cwts)/n
#     c = weight(b,1)
#     i = 1
#     U = r
#     for m in 1:n
#         while U > c
#             i += 1
#             c += weight(b, i)
#         end
#         U += last(cwts)/n
#         ps[m] = particles(b)[i]
#     end
#     return ParticleCollection(ps)
# 
# end
