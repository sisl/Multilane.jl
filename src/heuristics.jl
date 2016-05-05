# heuristics.jl
# heuristic policies

type Simple <: Policy
  mdp::NoCrashMDP
  A::NoCrashActionSpace
  sweeping_up::Bool
end
Simple(mdp::NoCrashMDP) = Simple(mdp,actions(mdp),true)

function action(p::Simple,s::MLState)
  goal_lane = p.mdp.rmodel.desired_lane
  y_desired = goal_lane
  dmodel = p.mdp.dmodel
  lc = sign(y_desired-s.env_cars[1].y) * dmodel.lane_change_vel
  acc = dmodel.adjustment_acceleration
  #if can't move towards desired lane sweep through accelerating and decelerating
  A = actions(p.mdp,s,p.A)
  A = A.NORMAL_ACTIONS[A.acceptable]
  if MLAction(0.,lc) in A
    return MLAction(0.,lc)
  end
  #also try faster or slower

  if (s.env_cars[1].vel >= dmode.phys_param.v_max) ||
      (s.env_cars[1].vel <= dmodel.phys_param.v_min)
    p.sweeping_up = !p.sweeping_up
  end

  # XXX does this work?
  if p.sweeping_up || !(MLAction(-acc,0.) in A)
    return MLAction(acc,0.)
  elseif !(MLAction(acc,0.) in A)
    return MLAction(-acc,0.)
  end
  return MLAction(0.,0.)
end
