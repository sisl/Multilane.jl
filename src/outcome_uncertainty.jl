"""
MDP with all state uncertainty modeled by outcome uncertainty
"""
struct OutcomeMDP{M} <: MDP{MLPhysicalState, MLAction}
    mdp::M
end

function generate_sr(m::OutcomeMDP, s::MLPhysicalState, a::MLAction, rng::AbstractRNG)
    cars = CarState[]
    for c in s.cars
        cs = CarState(c, rand(rng, m.mdp.dmodel.behaviors))
        push!(cars, cs)
    end
    mls = MLState(s, cars)
    mlsp, r = generate_sr(m.mdp, mls, a, rng)
    return MLPhysicalState(mlsp), r
end

actions(m::OutcomeMDP, s::MLPhysicalState) = actions(m.mdp, s)
discount(m::OutcomeMDP) = discount(m.mdp)

struct OutcomeSolver <: Solver
    solver::Solver
end

struct OutcomePlanner{P} <: Policy
    planner::P
end

function solve(sol::OutcomeSolver, mdp)
    omdp = OutcomeMDP(mdp)
    return OutcomePlanner(solve(sol.solver, omdp))
end

function action_info(p::OutcomePlanner, s::MLState)
    ps = MLPhysicalState(s)
    return action_info(p.planner, ps)
end
action(p::OutcomePlanner, s::MLState) = first(action_info(p, s))

Base.srand(p::OutcomePlanner, s) = srand(p.planner, s)
