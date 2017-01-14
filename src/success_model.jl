"""
Gives a reward of +1 if sp is the target lane
"""
type TargetLaneReward <: AbstractMLRewardModel
    target_lane::Int
end
function reward{D<:AbstractMLDynamicsModel}(mdp::MLMDP{MLState, MLAction, D, TargetLaneReward},
          s::MLState,
          ::MLAction,
          sp::MLState)
    return isnull(s.terminal) && sp.cars[1].y == mdp.rmodel.target_lane
end

function reward{D<:AbstractMLDynamicsModel}(mdp::MLPOMDP{MLState, MLAction, MLPhysicalState, D, TargetLaneReward},
          s::MLState,
          ::MLAction,
          sp::MLState)
    return isnull(s.terminal) && sp.cars[1].y == mdp.rmodel.target_lane
end
