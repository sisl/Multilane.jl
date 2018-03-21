"""
Gives a reward of +1 if sp is the target lane
"""
struct TargetLaneReward <: AbstractMLRewardModel
    target_lane::Int
end
function reward(mdp::MLMDP{MLState, MLAction, D, TargetLaneReward},
          s::MLState,
          ::MLAction,
          sp::MLState) where D<:AbstractMLDynamicsModel
    return isnull(s.terminal) && sp.cars[1].y == mdp.rmodel.target_lane
end

function reward(mdp::MLPOMDP{MLState, MLAction, MLPhysicalState, D, TargetLaneReward},
          s::MLState,
          ::MLAction,
          sp::MLState) where D<:AbstractMLDynamicsModel
    return isnull(s.terminal) && sp.cars[1].y == mdp.rmodel.target_lane
end

"""
Reward of +1 on transition INTO target lane, -lambda on unsafe transitions
"""
@with_kw struct SuccessReward <: AbstractMLRewardModel
    lambda::Float64 # always positive
    target_lane::Int
    brake_penalty_thresh::Float64 # always positive
end

function reward(p::NoCrashProblem{SuccessReward}, s::MLState, ::MLAction, sp::MLState)
    if sp.cars[1].y == p.rmodel.target_lane && s.cars[1].y != p.rmodel.target_lane
        r = 1.0
    else
        r = 0.0
    end
    nb_brakes = detect_braking(p, s, sp)
    r -= p.rmodel.lambda*nb_brakes
    return r
end
