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


struct BehaviorPOWNodeBelief{P<:MLPOMDP,G}
    model::P
    b::BehaviorParticleBelief{G}
    r::Float64
end

rand(rng::AbstractRNG, b::BehaviorPOWNodeBelief) = (rand(rng, b.b), b.r)

struct BehaviorPOWFilter
    params::WeightUpdateParams
end

POMCPOW.belief_type(::Type{BehaviorPOWFilter}, ::Type{P}) where P<:MLPOMDP = BehaviorPOWNodeBelief{P, gen_type(P)}

function POMCPOW.init_node_sr_belief(f::BehaviorPOWFilter, p::MLPOMDP, s, a, sp, o, r)
    particles = [Vector{IDMMOBILBehavior}() for c in o.cars]
    weights = [Vector{Float64}() for c in o.cars]
    gen = p.dmodel.behaviors
    maybe_push_one!(particles, weights, f.params, p.dmodel.phys_param, gen, sp, o)
    ab = BehaviorParticleBelief(gen, o, particles, weights)
    return BehaviorPOWNodeBelief(p, ab, r)
end

function POMCPOW.push_weighted!(b::BehaviorPOWNodeBelief, f::BehaviorPOWFilter, s, sp, r)
    maybe_push_one!(b.b.particles, b.b.weights, f.params, b.model.dmodel.phys_param, b.b.gen, sp, b.b.physical)
end

actions(p::MLPOMDP, b::BehaviorPOWNodeBelief) = actions(p, b.b.physical)
