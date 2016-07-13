# heuristics.jl
# heuristic policies

type Simple <: Policy #
  mdp::MDP
  A::NoCrashActionSpace
  sweeping_up::Bool
end
type SimpleSolver <: Solver end

Simple(mdp::MDP) = Simple(mdp,actions(mdp),true)
solve(solver::SimpleSolver, problem::MDP) = Simple(problem)
solve(solver::SimpleSolver, problem::EmbeddedBehaviorMDP) = Simple(problem.base)
POMDPs.updater(::Simple) = POMDPToolbox.FastPreviousObservationUpdater{MLObs}()
create_policy(s::SimpleSolver, problem::MDP) = Simple(problem)

function action(p::Simple,s::Union{MLState,MLObs},a::MLAction=create_action(p.mdp))
# lane changes if there is an opportunity
  goal_lane = p.mdp.rmodel.desired_lane
  y_desired = goal_lane
  dmodel = p.mdp.dmodel
  lc = sign(y_desired-s.env_cars[1].y) * dmodel.lane_change_rate
  acc = dmodel.adjustment_acceleration

  #if can't move towards desired lane sweep through accelerating and decelerating

  # TODO need an equivalent of is_safe that can operate on observations
  if is_safe(p.mdp, s, MLAction(0.,lc))
    return MLAction(0.,lc)
  end
  # maintain distance from other cars

  # maintain distance
  nbhd = get_neighborhood(dmodel.phys_param,s,1)

  if nbhd[2] == 0 && nbhd[5] == 0
    return MLAction(0.,0.)
  end

  dist_ahead = nbhd[2] != 0 ? s.env_cars[nbhd[2]].x - s.env_cars[1].x : Inf
  dist_behind = nbhd[5] != 0 ? s.env_cars[nbhd[5]].x - s.env_cars[1].x : Inf

  sgn = abs(dist_ahead) <= abs(dist_behind) ? -1 : 1

  accel = sgn * acc

  max_accel = max_safe_acc(p.mdp, s, 0.0)

  return MLAction(min(accel, max_accel),0.)
end

#=
Heuristics that I want:
Cautious - move over if it doesn't cause braking
Aggressive - move over if it doesn't cause a crash
=#

# type Simple <: Policy #
#   mdp::MDP
#   A::NoCrashActionSpace
#   sweeping_up::Bool
# end
# type SimpleSolver <: Solver end
# 
# Simple(mdp::MDP) = Simple(mdp,actions(mdp),true)
# solve(solver::SimpleSolver, problem::MDP) = Simple(problem)
# solve(solver::SimpleSolver, problem::EmbeddedBehaviorMDP) = Simple(problem.base)
# POMDPs.updater(::Simple) = POMDPToolbox.FastPreviousObservationUpdater{MLObs}()
# create_policy(s::SimpleSolver, problem::MDP) = Simple(problem)
# 
# function action(p::Simple,s::Union{MLState,MLObs},a::MLAction=create_action(p.mdp))
# # lane changes if there is an opportunity
#   goal_lane = p.mdp.rmodel.desired_lane
#   y_desired = goal_lane
#   dmodel = p.mdp.dmodel
#   lc = sign(y_desired-s.env_cars[1].y) * dmodel.lane_change_rate
#   acc = dmodel.adjustment_acceleration
# 
#   #if can't move towards desired lane sweep through accelerating and decelerating
# 
#   # TODO need an equivalent of is_safe that can operate on observations
#   if is_safe(p.mdp,s,MLAction(0.,lc))
#     return MLAction(0.,lc)
#   end
#   # maintain distance from other cars
# 
#   # maintain distance
#   nbhd = get_neighborhood(dmodel.phys_param,s,1)
# 
#   if nbhd[2] == 0 && nbhd[5] == 0
#     return MLAction(0.,0.)
#   end
# 
#   dist_ahead = nbhd[2] != 0 ? s.env_cars[nbhd[2]].x - s.env_cars[1].x : Inf
#   dist_behind = nbhd[5] != 0 ? s.env_cars[nbhd[5]].x - s.env_cars[1].x : Inf
# 
# 	sgn = abs(dist_ahead) <= abs(dist_behind) ? -1 : 1
# 
#   accel = sgn * acc
# 
#   return MLAction(accel,0.)
# end
