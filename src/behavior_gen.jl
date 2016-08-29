abstract BehaviorGenerator

type DiscreteBehaviorSet <: BehaviorGenerator
    models::Vector{BehaviorModel}
    weights::WeightVec
end

rand(rng::AbstractRNG, s::DiscreteBehaviorSet) = sample(rng, s.models, s.weights)

type UniformIDMMOBIL <: BehaviorGenerator
    min_idm::IDMParam
    max_idm::IDMParam
    min_mobil::MOBILParam
    max_mobil::MOBILParam
    next_idx::Int
end

function rand(rng::AbstractRNG, g::UniformIDMMOBIL)
    idm = IDMParam(
        g.min_idm.a + rand(rng)*(g.max_idm.a - g.min_idm.a),
        g.min_idm.b + rand(rng)*(g.max_idm.b - g.min_idm.b),
        g.min_idm.T + rand(rng)*(g.max_idm.T - g.min_idm.T),
        g.min_idm.v0 + rand(rng)*(g.max_idm.v0 - g.min_idm.v0),
        g.min_idm.s0 + rand(rng)*(g.max_idm.s0 - g.min_idm.s0),
        g.min_idm.del
    )
    mobil = MOBILParam(
        g.min_mobil.p + rand(rng)*(g.max_mobil.p - g.min_mobil.p),
        g.min_mobil.b_safe + rand(rng)*(g.max_mobil.b_safe - g.min_mobil.b_safe),
        g.min_mobil.a_thr + rand(rng)*(g.max_mobil.a_thr - g.min_mobil.a_thr),
    )
    g.next_idx += 1
    return IDMMOBILBehavior(idm, mobil, g.next_idx-1)
end

#=
N   T   A
120 100 140 v0  27.8 38.9   33.3    5.6
1.5 1.8 1.0 T   1.0 2.0     1.5     0.5
2.0 4.0 1.0 s0  0.0 4.0     2.0     2.0
1.4 1.0 2.0 a   0.8 2.0     1.4     0.6
2.0 1.0 3.0 b   1.0 3.0     2.0     1.0
=#

function standard_uniform(factor; correlated=false)
    ma = 1.4;    da = 0.6
    mb = 2.0;    db = 1.0
    mT = 1.5;    dT = 0.5
    mv0 = 33.3;  dv0 = 5.63
    ms0 = 2.0;   ds0 = 2.0
    del = 4.0
    mp = 0.5;    dp = 0.5
    mbsafe = 2.0;dbsafe = 1.0
    mathr = 0.1; dathr = 0.1
    max_idm = IDMParam(
        ma + factor*da,
        mb + factor*db,
        mT + factor*dT,
        mv0 + factor*dv0,
        ms0 + factor*ds0,
        del)
    min_idm = IDMParam(
        ma - factor*da,
        mb - factor*db,
        mT - factor*dT,
        mv0 - factor*dv0,
        ms0 - factor*ds0,
        del)
    max_mobil = MOBILParam(
        mp + factor*dp,
        mbsafe + factor*dbsafe,
        mathr + factor*dathr
    )
    min_mobil = MOBILParam(
        mp - factor*dp,
        mbsafe - factor*dbsafe,
        mathr - factor*dathr
    )
    if correlated
        return CorrelatedIDMMOBIL(min_idm, max_idm, min_mobil, max_mobil, 2)
    else
        return UniformIDMMOBIL(min_idm, max_idm, min_mobil, max_mobil, 2)
    end
end

type CorrelatedIDMMOBIL <: BehaviorGenerator
    min_idm::IDMParam
    max_idm::IDMParam
    min_mobil::MOBILParam
    max_mobil::MOBILParam
    next_idx::Int
end

function rand(rng::AbstractRNG, g::CorrelatedIDMMOBIL)
    agg = rand(rng)
    return create_model(g, agg)
end

function create_model(g::CorrelatedIDMMOBIL, agg::Float64)
    idm = IDMParam(
        g.min_idm.a + agg*(g.max_idm.a - g.min_idm.a),
        g.min_idm.b + agg*(g.max_idm.b - g.min_idm.b),
        g.min_idm.T + agg*(g.max_idm.T - g.min_idm.T),
        g.min_idm.v0 + agg*(g.max_idm.v0 - g.min_idm.v0),
        g.min_idm.s0 + agg*(g.max_idm.s0 - g.min_idm.s0),
        g.min_idm.del
    )
    mobil = MOBILParam(
        g.min_mobil.p + agg*(g.max_mobil.p - g.min_mobil.p),
        g.min_mobil.b_safe + agg*(g.max_mobil.b_safe - g.min_mobil.b_safe),
        g.min_mobil.a_thr + agg*(g.max_mobil.a_thr - g.min_mobil.a_thr),
    )
    g.next_idx += 1
    return IDMMOBILBehavior(idm, mobil, g.next_idx-1)
end

function aggressiveness(gen::CorrelatedIDMMOBIL, b::IDMMOBILBehavior)
    return (b.p_idm.v0 - gen.min_idm.v0)/(gen.max_idm.v0 - gen.min_idm.v0)
end

#=
function aggressiveness(gen::CorrelatedIDMMOBIL, b::IDMMOBILBehavior, tol)
    for n in fieldnames
end
=#
