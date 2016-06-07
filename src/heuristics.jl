# heuristics.jl
# heuristic policies

type Simple <: Policy
  mdp::Union{NoCrashMDP,NoCrashPOMDP}
  A::NoCrashActionSpace
  sweeping_up::Bool
end
type SimpleSolver <: Solver end

Simple(mdp::Union{NoCrashMDP,NoCrashPOMDP}) = Simple(mdp,actions(mdp),true)
solve(solver::SimpleSolver,problem::Union{NoCrashMDP,NoCrashPOMDP}) = Simple(problem)
POMDPs.updater(::Simple) = POMDPToolbox.FastPreviousObservationUpdater{MLObs}()

function action{MLState}(p::Simple,s::Union{MLState,MLObs},a::MLAction=create_action(p.mdp))
  goal_lane = p.mdp.rmodel.desired_lane
  y_desired = goal_lane
  dmodel = p.mdp.dmodel
  lc = sign(y_desired-s.env_cars[1].y) * dmodel.lane_change_vel / dmodel.phys_param.y_interval
  acc = dmodel.adjustment_acceleration

  #if can't move towards desired lane sweep through accelerating and decelerating

  # TODO need an equivalent of is_safe that can operate on observations
  if is_safe(p.mdp,s,MLAction(0.,lc))
    return MLAction(0.,lc)
  end
  # maintain distance from other cars

  #=
  if (s.env_cars[1].vel >= dmodel.phys_param.v_max) ||
      (s.env_cars[1].vel <= dmodel.phys_param.v_min)
    p.sweeping_up = !p.sweeping_up
  end

  # XXX does this work?
  if p.sweeping_up || !(MLAction(-acc,0.) in A)
    return MLAction(acc,0.)
  elseif !(MLAction(acc,0.) in A)
    return MLAction(-acc,0.)
  end
  =#
  # maintain distance
  nbhd = get_neighborhood(dmodel.phys_param,s,1)

  if nbhd[2] == 0 && nbhd[5] == 0
    return MLAction(0.,0.)
  end

  dist_ahead = nbhd[2] != 0 ? s.env_cars[nbhd[2]].x - s.env_cars[1].x : Inf
  dist_behind = nbhd[5] != 0 ? s.env_cars[nbhd[5]].x - s.env_cars[1].x : Inf

	sgn = abs(dist_ahead) <= abs(dist_behind) ? -1 : 1

  accel = sgn * acc

  return MLAction(accel,0.)
end
