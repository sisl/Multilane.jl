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
    lambda::Float64                 = 1.0  # always positive
    target_lane::Int                = 4
    brake_penalty_thresh::Float64   = 4.0  # always positive
    speed_thresh::Float64           = 15.0 # always positive
end

function reward(p::NoCrashProblem{SuccessReward}, s::MLState, ::MLAction, sp::MLState)
    if sp.cars[1].y == p.rmodel.target_lane && s.cars[1].y != p.rmodel.target_lane
        r = 1.0
    else
        r = 0.0
    end
    min_speed = minimum(c.vel for c in sp.cars)
    nb_brakes = detect_braking(p, s, sp)
    if nb_brakes > 0 || min_speed < p.rmodel.speed_thresh
        r -= p.rmodel.lambda
    end
    return r
end
