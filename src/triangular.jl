# A method of rand to sample from a triangular distribution
# https://en.wikipedia.org/wiki/Triangular_distribution

import Distributions.TriangularDist
import Distributions.params

function rand(rng::AbstractRNG, d::TriangularDist)
    (a, b, c) = params(d)
    b_m_a = b - a
    u = rand(rng)
    if b_m_a * u < (c - a)
        return a + sqrt(u * b_m_a * (c - a))
    else
        return b - sqrt((1 - u) * b_m_a * (b - c))
    end
end
