type ParticleUpdater <: POMDPs.Updater{ParticleBelief{MLState}}
  nb_particles::Int
  problem::NoCrashPOMDP
  rng::AbstractRNG
end

create_belief(u::ParticleUpdater) = ParticleBelief{MLState}(Particle{MLState}[Particle{MLState}(create_state(u.problem),1.) for _ = 1:u.nb_particles], Dict{MLState,Float64}(), Float64[], false)

create_belief(u::ParticleUpdater, s::MLState) = ParticleBelief{MLState}(Particle{MLState}[Particle{MLState}(deepcopy(s),1.) for _ = 1:u.nb_particles], Dict{MLState,Float64}(), Float64[], false)

function obs_to_state(mdp::NoCrashPOMDP,o::MLObs, s::MLState)
  env_cars = CarState[CarState(c.x,c.y,c.vel,c.lane_change,b.behavior,c.id) for (c,b) in zip(o.env_cars, s.env_cars)]
  return MLState(false, env_cars)
  #return MLState(false,CarState[CarState(c.x,c.y,c.vel,c.lane_change,i==1?Nullable{BehaviorModel}():sample(mdp.dmodel.behaviors), c.id) for (i,c) in enumerate(o.env_cars)])
end

function similarity(pomdp::NoCrashPOMDP, sp::MLState, o::MLObs)
  # If a car goes out of bounds: no similarity

  if sp.crashed != o.crashed
    return 0.
  end

  s_cars = Dict{Int,CarState}([car.id => car for car in sp.env_cars])
  o_cars = Dict{Int,CarStateObs}([car.id => car for car in o.env_cars])

  if length(intersect(keys(s_cars),keys(o_cars))) != length(s_cars)
    return 0.
  end

  dist = 0.

  for id in keys(s_cars)
    s_car = s_cars[id]
    o_car = o_cars[id]

    s = [s_car.x; s_car.y; s_car.vel; s_car.lane_change]
    o = [o_car.x; o_car.y; o_car.vel; o_car.lane_change]

    dist += norm(s-o)
  end

  return 1./(0.1 + dist) #or something else

end

function POMDPs.update(updater::ParticleUpdater, belief_old::ParticleBelief, a::MLAction, o::MLObs, belief_new::ParticleBelief=create_belief(updater))

  states = [p.state for p in belief_old.particles]
  wv = WeightVec([p.weight for p in belief_old.particles])
  for i = 1:updater.nb_particles
    s = sample(updater.rng,states, wv)
    sp, r = Multilane.generate_sr(updater.problem, s, a, updater.rng)
    """
    alt: set sp to o w/ random behavior models
    """
    # check how close the propagated model is to the actual thing
    w = similarity(updater.problem, sp, o) #p(o | s', a) # TODO how??? ABC?
    sp = obs_to_state(updater.problem, o, s)
    belief_new.particles[i] = Particle(sp, w)
  end

  return belief_new
end

rand(rng::AbstractRNG, b::ParticleBelief{MLState},s::MLState) = sample([p.state for p in b.particles], WeightVec([p.weight for p in b.particles]))
rand(rng::AbstractRNG, b::ParticleBelief{MLState}) = sample([p.state for p in b.particles], WeightVec([p.weight for p in b.particles]))

initialize_belief(u::ParticleUpdater, db::AbstractDistribution) = begin db end #do nothing XXX TODO

initialize_belief(u::ParticleUpdater, db::AbstractDistribution, b::ParticleBelief{MLState}) = begin db end #do nothing XXX TODO


function actions(pomdp::NoCrashPOMDP, b::ParticleBelief{MLState}, as::NoCrashActionSpace=actions(pomdp))
  # XXX temp
  _as = actions(pomdp)
  as = actions(pomdp) #XXX inefficient? need to reset anyways
  acceptable = as.acceptable
  brake = 0.
  for particle in b.particles
    _as = actions(pomdp, particle.state, _as)
    acceptable = intersect(as.acceptable, _as.acceptable)
    brake = min(_as.brake.acc, brake)
  end
  as = NoCrashActionSpace(as.NORMAL_ACTIONS, acceptable, MLAction(min(brake, -pomdp.rmodel.dangerous_brake_threshold/2.0), 0.))
  return as
end


# TODO clean up
#=
import DESPOT: fringe_upper_bound, lower_bound, init_lower_bound

function fringe_upper_bound(pomdp::NoCrashPOMDP, s::MLState)
  """
  Assume there are no cars in the environment: just go straight to target lane, and stay there
    for the rest of the time horizon
  """
  # NOTE: This function makes assumptions about the length of the time horizon!!!!
  T = 100 #time horizon
  pp = pomdp.dmodel.phys_param
  nb_lanes = pp.nb_lanes
  current_lane = s.env_cars[1].y
  t_single_lane_change = pp.w_lane / (pp.dt * pomdp.dmodel.lane_change_vel) #nb timesteps
  t_target_lane = (nb_lanes - current_lane) * t_single_lane_change
  return sum([pomdp.rmodel.reward_in_desired_lane * discount(pomdp)^t for t = 0:(T-t_target_lane-1)])
end

type NoCrashLB <: DESPOTLowerBound end

init_lower_bound(lb::NoCrashLB, pomdp::NoCrashPOMDP, config::DESPOTConfig) = begin end

function lower_bound(lb::NoCrashLB, pomdp::NoCrashPOMDP, particles::Vector{DESPOTParticle{MLState}}, config:DESPOTConfig)
  """
  Just rollout the heuristic policy--it only really depends on observations, so
    bmodel (eg true behavior states) don't matter
  """

  s = particles[1].state

  sim = RolloutSimulator(initial_state=s, max_steps=100)

  policy = Simple(pomdp) # TODO check if this signature works with NoCrashPOMDP and stuff

  return simulate(sim, pomdp, policy)
end
=#
