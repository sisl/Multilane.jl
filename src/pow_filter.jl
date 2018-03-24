struct AggressivenessPOWNodeBelief{P<:MLPOMDP}
    model::P
    b::AggressivenessBelief
    r::Float64
end

rand(rng::AbstractRNG, b::AggressivenessPOWNodeBelief) = (rand(rng, b.b), b.r)

struct AggressivenessPOWFilter end

belief_type(filter::AggressivenessPOWFilter, pomdp::MLPOMDP) = AggressivenessPOWNodeBelief{typeof(pomdp)}

function init_node_sr_belief(filter::AggressivenessPOWFilter, p::MLPOMDP, s, a, sp, o, r)
    particles = Vector{Float64}[]
    weights = Vector{Float64}[]
    push_one!(particles, weights, p.dmodel.phys_param, p.dmodel.behaviors, sp, o)
    ab = AggressivenessBelief(gen, o, particles, weights)
    return AggressivenessPOWNodeBelief(p, ab, r)
end

function push_wieghted!(b::AggressivenessPOWNodeBelief, ::AggressivenessPOWFilter, s, sp, r)
    maybe_push_one!(b.b.particles, b.b.weights, b.model.dmodel.phys_param, b.b.gen, sp, o)
end
