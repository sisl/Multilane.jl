type ParticleUpdater <: POMDPs.Updater{ParticleBelief{MLState}}
  nb_particles::Int
  problem::NoCrashPOMDP
  rng::AbstractRNG
end

create_belief(u::ParticleUpdater) = ParticleBelief{MLState}(Particle{MLState}[Particle{MLState}(create_state(u.problem),1.) for _ = 1:u.nb_particles])

create_belief(u::ParticleUpdater, s::MLState) = ParticleBelief{MLState}(Particle{MLState}[Particle{MLState}(s,1.) for _ = 1:u.nb_particles])

function MLState(mdp::NoCrashMDP,o::MLObs)
  return MLState(false,CarState[CarState(c.x,c.y,c.vel,c.lane_change,i==1?Nullable{BehaviorModel}():sample(mdp.dmodel.BEHAVIORS)) for (i,c) in enumerate(o.env_cars)])
end

function POMDPs.update(updater::ParticleUpdater, belief_old::ParticleBelief, a::MLAction, o::MLObs, belief_new::ParticleBelief=create_belief(updater))

  for i = 1:updater.nb_particles
    # XXX this is probably SUPER inefficient
    #s = sample(updater.rng, [p.state for p in belief_old.particles], WeightVec([[p.weight for p in belief_old.particles]]))
    #sp, r = Multilane.generate_sr(updater.problem, s, a, updater.rng)
    """
    alt: set sp to o w/ random behavior models
    """
    sp = MLState(updater.problem, o)
    w = 1. #p(o | s', a) # TODO
    belief_new.particles[i] = Particle(sp, w)
  end

  return belief_new
end

rand(rng::AbstractRNG, b::ParticleBelief{MLState},s::MLState) = sample([p.state for p in b.particles], WeightVec([p.weight for p in b.particles]))
rand(rng::AbstractRNG, b::ParticleBelief{MLState}) = sample([p.state for p in b.particles], WeightVec([p.weight for p in b.particles]))

initialize_belief(u::ParticleUpdater, b::ParticleBelief{MLState}) = begin b end #do nothing XXX TODO

initialize_belief(u::ParticleUpdater, __::AbstractDistribution, b::ParticleBelief{MLState}) = begin b end #do nothing XXX TODO
