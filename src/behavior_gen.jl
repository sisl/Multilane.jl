abstract type BehaviorGenerator end

mutable struct DiscreteBehaviorSet <: BehaviorGenerator
    models::Vector{BehaviorModel}
    weights::Weights
end

rand(rng::AbstractRNG, s::DiscreteBehaviorSet) = sample(rng, s.models, s.weights)

function max_accel(gen::DiscreteBehaviorSet)
    m = gen.models
    w = gen.weights
    len = length(m)
    return maximum(max_accel(m[i]) for i in 1:len if w[i] > 0.0)
end

mutable struct UniformIDMMOBIL <: BehaviorGenerator
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
1.5 1.8 1.0 T   1.0  2.0     1.5     0.5
2.0 4.0 1.0 s0  0.0  4.0     2.0     2.0
1.4 1.0 2.0 a   0.8  2.0     1.4     0.6
2.0 1.0 3.0 b   1.0  3.0     2.0     1.0
=#

function standard_uniform(factor=1.0; correlation::Union{Bool,Float64}=false)
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
                       max(ma - factor*da, 0.01),
                       max(mb - factor*db, 0.01),
                       max(mT - factor*dT, 0.0),
                       max(mv0 - factor*dv0, 0.01),
                       max(ms0 - factor*ds0, 0.0),
                       del)
    max_mobil = MOBILParam(
        mp + factor*dp,
        mbsafe + factor*dbsafe,
        mathr + factor*dathr
    )
    min_mobil = MOBILParam(
                           max(mp - factor*dp, 0.0),
                           max(mbsafe - factor*dbsafe, 0.01),
                           max(mathr - factor*dathr, 0.0)
    )
    if correlation == true
        return CorrelatedIDMMOBIL(min_idm, max_idm, min_mobil, max_mobil, 2)
    elseif correlation == false
        return UniformIDMMOBIL(min_idm, max_idm, min_mobil, max_mobil, 2)
    else
        return CopulaIDMMOBIL(min_idm, max_idm, min_mobil, max_mobil, correlation)
    end
end


mutable struct CorrelatedIDMMOBIL <: BehaviorGenerator
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
        g.max_idm.T + agg*(g.min_idm.T - g.max_idm.T), # T is lower for more aggressive
        g.min_idm.v0 + agg*(g.max_idm.v0 - g.min_idm.v0),
        g.max_idm.s0 + agg*(g.min_idm.s0 - g.max_idm.s0), # s0 is lower for more aggressive
        g.min_idm.del
    )
    mobil = MOBILParam(
        g.max_mobil.p + agg*(g.min_mobil.p - g.max_mobil.p), # p is lower for more aggressive
        g.min_mobil.b_safe + agg*(g.max_mobil.b_safe - g.min_mobil.b_safe),
        g.max_mobil.a_thr + agg*(g.min_mobil.a_thr - g.max_mobil.a_thr), # a_thr is lower for more aggressive
    )
    g.next_idx += 1
    return IDMMOBILBehavior(idm, mobil, g.next_idx-1)
end

function aggressiveness(gen::CorrelatedIDMMOBIL, b::IDMMOBILBehavior)
    return (b.p_idm.v0 - gen.min_idm.v0)/(gen.max_idm.v0 - gen.min_idm.v0)
end

mutable struct CopulaIDMMOBIL <: BehaviorGenerator
    min_idm::IDMParam
    max_idm::IDMParam
    min_mobil::MOBILParam
    max_mobil::MOBILParam
    copula::GaussianCopula
    next_idx::Int
end

function CopulaIDMMOBIL(min_idm::IDMParam,
                        max_idm::IDMParam,
                        min_mobil::MOBILParam,
                        max_mobil::MOBILParam,
                        cor::Float64)
    return CopulaIDMMOBIL(min_idm, max_idm,
                          min_mobil, max_mobil,
                          GaussianCopula(8, cor), 2)
end

function rand(rng::AbstractRNG, g::CopulaIDMMOBIL)
    agg = rand(rng, g.copula)
    return create_model(g, agg)
end

function create_model(g::CopulaIDMMOBIL, agg::Vector{Float64})
    @assert length(agg) == 8
    idm = IDMParam(
        g.min_idm.a + agg[1]*(g.max_idm.a - g.min_idm.a),
        g.min_idm.b + agg[2]*(g.max_idm.b - g.min_idm.b),
        g.max_idm.T + agg[3]*(g.min_idm.T - g.max_idm.T), # T is lower for more aggressive
        g.min_idm.v0 + agg[4]*(g.max_idm.v0 - g.min_idm.v0),
        g.max_idm.s0 + agg[5]*(g.min_idm.s0 - g.max_idm.s0), # s0 is lower for more aggressive
        g.min_idm.del
    )
    mobil = MOBILParam(
        g.max_mobil.p + agg[6]*(g.min_mobil.p - g.max_mobil.p), # p is lower for more aggressive
        g.min_mobil.b_safe + agg[7]*(g.max_mobil.b_safe - g.min_mobil.b_safe),
        g.max_mobil.a_thr + agg[8]*(g.min_mobil.a_thr - g.max_mobil.a_thr), # a_thr is lower for more aggressive
    )
    g.next_idx += 1
    return IDMMOBILBehavior(idm, mobil, g.next_idx-1)
end

CorrelatedIDMMOBIL(gen::Union{CopulaIDMMOBIL, UniformIDMMOBIL}) = CorrelatedIDMMOBIL(
    gen.min_idm,
    gen.max_idm,
    gen.min_mobil,
    gen.max_mobil,
    gen.next_idx
)

max_accel(gen::BehaviorGenerator) = 1.5*gen.max_idm.a

function clip(b::IDMMOBILBehavior, gen::Union{UniformIDMMOBIL,CopulaIDMMOBIL})
    bi = b.p_idm
    bm = b.p_mobil
    mini = gen.min_idm
    maxi = gen.max_idm
    minm = gen.min_mobil
    maxm = gen.max_mobil
    return IDMMOBILBehavior(
        IDMParam(
            max(min(bi.a, maxi.a), mini.a),
            max(min(bi.b, maxi.b), mini.b),
            max(min(bi.T, maxi.T), mini.T),
            max(min(bi.v0, maxi.v0), mini.v0),
            max(min(bi.s0, maxi.s0), mini.s0),
            max(min(bi.del, maxi.del), mini.del)
        ),
        MOBILParam(
            max(min(bm.p, maxm.p), minm.p),
            max(min(bm.b_safe, maxm.b_safe), minm.b_safe),
            max(min(bm.a_thr, maxm.a_thr), minm.a_thr)
        ),
        b.idx
    )
end

const STANDARD_CORRELATED = standard_uniform(correlation=true)

function normalized_error_sum(a::IDMMOBILBehavior, b::IDMMOBILBehavior, gen::BehaviorGenerator)
    return sum(abs.(a.p_idm-b.p_idm)./(gen.max_idm-gen.min_idm+1e-5)) + # the 1e-5 is to protect from div by zero
           sum(abs.(a.p_mobil-b.p_mobil)./(gen.max_mobil-gen.min_mobil))
end
