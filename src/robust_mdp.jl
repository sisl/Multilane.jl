type RobustNoCrashMDP <: RobustMDP{MLPhysicalState, MLAction}
    base::MLMDP
end
RobustMCTS.representative_mdp(rmdp::RobustNoCrashMDP) = StochasticBehaviorNoCrashMDP(rmdp.base)

function RobustMCTS.next_model(gen::RandomModelGenerator, rmdp::RobustNoCrashMDP, s::MLPhysicalState, a::MLAction)
    behaviors = Dict{Int, Nullable{BehaviorModel}}(1=>nothing)
    for c in s.env_cars
        # TODO instead of just using sample, try to pick models that will be bad
        behaviors[c.id] = rand(gen.rng, rmdp.base.dmodel.behaviors)
    end
    return FixedBehaviorNoCrashMDP(behaviors, rmdp.base)
end

abstract EmbeddedBehaviorMDP <: MDP{MLPhysicalState, MLAction}

actions(p::EmbeddedBehaviorMDP) = actions(p.base)
actions(p::EmbeddedBehaviorMDP, s::MLPhysicalState, as::NoCrashActionSpace) = actions(p.base, s, as)
create_action(p::EmbeddedBehaviorMDP) = create_action(p.base)
create_state(p::EmbeddedBehaviorMDP) = MLPhysicalState(false, [])
discount(p::EmbeddedBehaviorMDP) = discount(p.base)
# reward(p::EmbeddedBehaviorMDP, s::MLPhysicalState, a::MLAction, sp::MLPhysicalState)

type FixedBehaviorNoCrashMDP <: EmbeddedBehaviorMDP
    behaviors::Dict{Int,Nullable{BehaviorModel}}
    base::MLMDP
end

function generate_sr(mdp::FixedBehaviorNoCrashMDP, s::MLPhysicalState, a::MLAction, rng::AbstractRNG, sp::MLPhysicalState=create_state(mdp))
    full_s = MLState(s.crashed, Array(CarState, length(s.env_cars)))
    for (i,c) in enumerate(s.env_cars) 
        full_s.env_cars[i] = CarState(c, mdp.behaviors[c.id])
    end
    full_sp = generate_s(mdp.base, full_s, a, rng)
    for c in sp.env_cars
        if !haskey(mdp.behaviors, c.id)
            mdp.behaviors[c.id] = c.behavior
        end
    end
    return MLPhysicalState(full_sp), reward(mdp.base, full_s, a, full_sp)
end

type StochasticBehaviorNoCrashMDP <: EmbeddedBehaviorMDP
    base::MLMDP
end

function generate_sr(mdp::StochasticBehaviorNoCrashMDP, s::MLPhysicalState, a::MLAction, rng::AbstractRNG, sp::MLPhysicalState=create_state(mdp))
    full_s = MLState(s.crashed, Array(CarState, length(s.env_cars)))
    for i in 1:length(s.env_cars) 
        full_s.env_cars[i] = CarState(s.env_cars[i], rand(rng, mdp.base.dmodel.behaviors))
    end
    full_sp = generate_s(mdp.base, full_s, a, rng)
    return MLPhysicalState(full_sp), reward(mdp.base, full_s, a, full_sp)
end

function initial_state(mdp::Union{FixedBehaviorNoCrashMDP,StochasticBehaviorNoCrashMDP}, rng::AbstractRNG, s=nothing)
    full_state = initial_state(mdp.base, rng::AbstractRNG)
    return MLPhysicalState(full_state)
end

type RobustMLSolver <: Solver
    rsolver
end

type RobustMLPolicy <: Policy{MLState}
    rpolicy
end

function action(p::RobustMLPolicy, s::MLState, a::MLAction=MLAction(0,0))
    mdp = representative_mdp(p.rpolicy.rmdp)
    as = actions(mdp, MLPhysicalState(s), actions(mdp))
    if length(as) == 1
        return collect(as)[1]
    end
    action(p.rpolicy, MLPhysicalState(s))
end

function solve(s::RobustMLSolver, mdp::MLMDP)
    RobustMLPolicy(solve(s.rsolver, RobustNoCrashMDP(mdp::MLMDP)))
end
