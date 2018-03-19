"""
Gives a reward of +1 if sp is the target lane
"""
mutable struct TargetLaneReward <: AbstractMLRewardModel
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
