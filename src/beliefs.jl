type DiscreteBehaviorBelief <: AbstractDistribution
    ps::MLPhysicalState
    models::AbstractVector
    weights::Vector{Vector{Float64}}
end
DiscreteBehaviorBelief(ps::MLPhysicalState, models::AbstractVector) = DiscreteBehaviorBelief(ps, models, Array(Vector{Float64}, length(ps.env_cars)))

function rand(rng::AbstractRNG,
              b::DiscreteBehaviorBelief,
              s::MLState=MLState(b.ps.crashed, Array(CarState, length(b.ps.env_cars))))
    resize!(s.env_cars, length(b.ps.env_cars))
    for i in 1:length(s.env_cars)
        s.env_cars[i] = CarState(b.ps.env_cars[i], sample(rng, b.models, WeightVec(b.weights[i])))
    end
    return s
end

@with_kw type WeightUpdateParams
    smoothing::Float64 = 0.02 # value between 0 and 1, adds this fraction of the max to each entry in the vecot
    wrong_lane_factor::Float64 = 0.1
end

function weights_from_particles!(b::DiscreteBehaviorBelief, problem::NoCrashProblem, o::MLPhysicalState, particles, p::WeightUpdateParams)
    b.ps = o
    resize!(b.weights, length(o.env_cars))
    for i in 1:length(o.env_cars)
        b.weights[i] = zeros(length(b.models)) #XXX lots of allocation
    end
    for sp in particles
        isp = 1
        io = 1
        while io <= length(o.env_cars) && isp <= length(sp.env_cars)
            co = o.env_cars[io]
            csp = sp.env_cars[isp]
            if co.id == csp.id
                # sigma_acc = dmodel.vel_sigma/dt
                # dv = acc*dt
                # sigma_v = sigma_dv = dmodel.vel_sigma
                if abs(co.x-csp.x) < 0.2*problem.dmodel.phys_param.lane_length
                    proportional_likelihood = proportional_normal_pdf(csp.vel,
                                                                      co.vel,
                                                                      problem.dmodel.vel_sigma)
                    if co.y == csp.y
                        b.weights[io][get(csp.behavior).idx] += proportional_likelihood
                    elseif abs(co.y - csp.y) < 1.0
                        b.weights[io][get(csp.behavior).idx] += p.wrong_lane_factor*proportional_likelihood
                    end # if greater than one lane apart, do nothing
                end
                io += 1
                isp += 1
            elseif co.id < csp.id
                io += 1
            else 
                @assert co.id > csp.id
                isp += 1
            end
        end
    end
    
    # smoothing
    for i in 1:length(b.weights)
        if sum(b.weights[i]) > 0.0
            b.weights[i] .+= p.smoothing * sum(b.weights[i])
        else
            b.weights[i] .+= 1.0
        end
    end

    return b
end

type BehaviorBeliefUpdaterStub <: Updater{DiscreteBehaviorBelief}
    smoothing::Float64
    nb_particles::Int
end

type BehaviorBeliefUpdater <: Updater{DiscreteBehaviorBelief}
    problem::NoCrashProblem
    smoothing::Float64
    nb_particles::Float64
end

function update(up::BehaviorBeliefUpdater,
                b_old::DiscreteBehaviorBelief,
                a::MLAction,
                o::MLPhysicalState,
                b_new::DiscreteBehaviorBelief=DiscreteBehaviorBelief(o,
                                                       up.problem.dmodel.behaviors,
                                                       Array(Vector{Float64}, 0)))
    # particles = SharedArray(MLState, )
    # @parallel
    # # run simulations

    # weights_from_particles!(b_new, )
    error("not implemented")
end

type BehaviorRootUpdaterStub <: Updater
    params::WeightUpdateParams
end
 
type BehaviorRootUpdater <: Updater{POMCP.BeliefNode}
    problem::NoCrashProblem
    params::WeightUpdateParams
end

initialize_belief(up::BehaviorRootUpdater, b) = POMCP.RootNode(b)

function update(up::BehaviorRootUpdater,
                b_old::POMCP.BeliefNode,
                a::MLAction,
                o::MLPhysicalState,
                b_new::POMCP.RootNode=POMCP.RootNode(DiscreteBehaviorBelief(o, up.problem.dmodel.behaviors.models)))

    b_new.B.models = up.problem.dmodel.behaviors.models
    #XXX hack
    particles = Iterators.chain([child.B.particles for child in values(b_old.children[a].children)]...)

    weights_from_particles!(b_new.B, up.problem, o, particles, up.params)
    return b_new
end

proportional_normal_pdf(x, mu, sigma) = exp(-(x-mu)^2/(2.*sigma^2))


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
  t_single_lane_change = 1.0 / (pp.dt * pomdp.dmodel.lane_change_rate) #nb timesteps
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
