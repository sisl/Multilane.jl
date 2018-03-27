# redundant
struct QMDPState{B,S}
    isstate::Bool
    b::Nullable{B}
    s::Nullable{S}

    function QMDPState{B,S}(isstate::Bool, x) where {B,S}
        if isstate
            return new(isstate, Nullable{B}(), Nullable(x))
        else
            return new(isstate, Nullable(x), Nullable{S}())
        end
    end
end

struct QMDPWrapper{M,B,S,A} <: MDP{QMDPState{B,S}, A}
    mdp::M
end

function QMDPWrapper(mdp::MDP, B::Type)
    S = state_type(mdp)
    A = POMDPs.action_type(mdp)
    return QMDPWrapper{typeof(mdp), B, S, A}(mdp)
end

state(m::QMDPWrapper{<:Any,B,S,<:Any}, s::S) where {B,S} = QMDPState{B,S}(true, s)
state(m::QMDPWrapper{<:Any,B,S,<:Any}, b::B) where {B,S} = QMDPState{B,S}(false, b)

function isterminal(m::QMDPWrapper, s::QMDPState)
    if s.isstate
        return isterminal(m, get(s.s))
    else
        return isterminal(m, get(s.b))
    end
end

function generate_sr(m::QMDPWrapper, s::QMDPState, a, rng::AbstractRNG)
    if s.isstate
        sp, r = generate_sr(m.mdp, get(s.s), a, rng)
        return state_type(m)(true, sp), r
    else
        sp, r = generate_sr(m.mdp, rand(rng, get(s.b)), a, rng)
        return state_type(m)(true, sp), r
    end
end

discount(m::QMDPWrapper) = discount(m.mdp)
function actions(m::QMDPWrapper, s::QMDPState)
    if s.isstate
        return actions(m.mdp, get(s.s))
    else
        # Gallium.@enter actions(m.mdp, get(s.b))
        return actions(m.mdp, get(s.b))
    end
end

struct GenQMDPSolver <: Solver
    solver
end

struct GenQMDPPolicy{P<:Policy, Q<:QMDPWrapper} <: Policy
    policy::P
    qmdp::Q
end

solve(sol::GenQMDPSolver, qmdp::QMDPWrapper) = GenQMDPPolicy(solve(sol.solver, qmdp), qmdp)

function action_info(p::GenQMDPPolicy, b)
    # XXX if this ever makes it into the toolbox, need to get the mdp some other way
    s = QMDPState{typeof(b), state_type(p.qmdp.mdp)}(false, b)
    return action_info(p.policy, s)
end

action(p::GenQMDPPolicy, b) = first(action_info(p, b))
