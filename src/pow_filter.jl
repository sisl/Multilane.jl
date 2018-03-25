struct AggressivenessPOWNodeBelief{P<:MLPOMDP}
    model::P
    b::AggressivenessBelief
    r::Float64
end

rand(rng::AbstractRNG, b::AggressivenessPOWNodeBelief) = (rand(rng, b.b), b.r)

struct AggressivenessPOWFilter
    params::WeightUpdateParams
end

POMCPOW.belief_type(::Type{AggressivenessPOWFilter}, ::Type{P}) where P<:MLPOMDP = AggressivenessPOWNodeBelief{P}

function POMCPOW.init_node_sr_belief(f::AggressivenessPOWFilter, p::MLPOMDP, s, a, sp, o, r)
    particles = [Vector{Float64}() for c in o.cars]
    weights = [Vector{Float64}() for c in o.cars]
    gen = CorrelatedIDMMOBIL(p.dmodel.behaviors)
    maybe_push_one!(particles, weights, f.params, p.dmodel.phys_param, gen, sp, o)
    ab = AggressivenessBelief(gen, o, particles, weights)
    return AggressivenessPOWNodeBelief(p, ab, r)
end

function POMCPOW.push_weighted!(b::AggressivenessPOWNodeBelief, f::AggressivenessPOWFilter, s, sp, r)
    maybe_push_one!(b.b.particles, b.b.weights, f.params, b.model.dmodel.phys_param, b.b.gen, sp, b.b.physical)
end

actions(p::MLPOMDP, b::AggressivenessPOWNodeBelief) = actions(p, b.b.physical)

function actions(p::MLPOMDP, h::POWTreeObsNode) 
    if isroot(h)
        return actions(p, belief(h))
    else
        return actions(p, sr_belief(h))
    end
end
