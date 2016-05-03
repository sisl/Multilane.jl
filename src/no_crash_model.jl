type NoCrashRewardModel <: AbstractMLRewardModel
    cost_emergency_brake::Float64
    reward_in_desired_lane::Float64

    dangerous_brake_threshold::Float64 # if the deceleration is greater than this cost_emergency_brake will be accured
    desired_lane::Int
end

type NoCrashIDMMOBILModel <: AbstractMLDynamicsModel
    nb_cars::Int
    phys_param::PhysicalParam

    behaviors::Vector{(Symbol, BehaviorModel)}
    behavior_probabilities::WeightVector

    adjustment_acceleration::Float64
    lane_change_vel::Float64

    p_appear::Float64 # probability of a new car appearing if the maximum number are not on the road
    appear_clearance::Float64 # minimum clearance for a car to appear

    vel_sigma::Float64 # std of new car speed about v0
    lane_weights::Array{Float64,1} # dirichlet alpha values for each lane: first is for rightmost lane
    dist_lambda::Float64 # exponential distr param for first car's distance from top of lane
end

typealias NoCrashMDP MLMDP{MLState, MLAction, NoCrashIDMMOBILModel, NoCrashRewardModel}

# action space = {a in {accelerate,maintain,decelerate}x{left_lane_change,maintain,right_lane_change} | a is safe} U {EMERGENCY_BRAKE}
immutable NoCrashActionSpace
    NORMAL_ACTIONS::Vector{MLAction} # all the actions except emergency brake
    acceptable::IntSet
    emergency_brake::MLAction
end
# TODO for performance, make this a macro?
const NB_NORMAL_ACTIONS::Int = 9

function NoCrashActionSpace(mdp::NoCrashMDP)
    accels = (-mdp.adjustment_acceleration, 0.0, mdp.adjustment_acceleration)
    lane_changes = (-1, 0 1)
    NORMAL_ACTIONS = MLAction[MLAction(a,l) for (a,l) in Iterators.product(accels, lane_changes)]
    return NoCrashActionSpace(NORMAL_ACTIONS, IntSet(), MLAction()) # note: emergency brake will be calculated later based on the state
end

function actions(mdp::NoCrashMDP)
    return NoCrashActionSpace(mdp)
end

function actions(mdp::NoCrashMDP, s::MLState, as::NoCrashActionSpace) # no implementation without the third arg to enforce efficiency
    acceptable = IntSet()
    for i in 1:NB_NORMAL_ACTIONS
        if !is_crash(mdp, s, as.NORMAL_ACTIONS[i]) # TODO: Make this faster by doing it all at once and saving some calculations
            push!(acceptable, i)
        end
    end
    emergency_brake = MLAction(e_brake_acc(mdp, s), 0)
    return NoCrashActionSpace(as.NORMAL_ACTIONS, acceptable, emergency_brake)
end

iterator(as::NoCrashActionSpace) = as
Base.start(as::NoCrashActionSpace) = 1
function Base.next(as::NoCrashActionSpace, state::Integer)
    while !(state in as)
        if state > NB_NORMAL_ACTIONS
            return (s.emergency_brake, state+1)
        end
        state += 1
    end
    return (as.NORMAL_ACTIONS[state], state+1)
end
Base.done(as::NoCrashActionSpace, state::Integer) = state > NB_NORMAL_ACTIONS+1

function rand(rng::AbstractRNG, as::NoCrashActionSpace, a::MLAction=MLAction())
    nb_acts = length(as.acceptable)+1
    index = rand(rng,1:nb_acts)
    for (i,a) in enumerate(as) # is this efficient at all ?
        if i == index
            return a
        end
    end
end

"""
Calculates the emergency braking acceleration
"""
function e_brake_acc(mdp::NoCrashMDP, s::MLState)
    # find car immediately in front
    # calculate necessary acceleration
end

function generate_sr(mdp::NoCrashMDP, s::MLState, a::MLAction, rng::AbstractRNG, sp::MLState=create_state(mdp))

    pp = mdp.dmodel.phys_param
    dt = pp.dt
    nb_cars = length(s.env_cars)
    resize!(sp.env_cars, nb_cars)
    r = 0.0 # reward

    sp.env_cars[1].lane_change = a.lane_change
    ## Calculate deltas ##
    #====================#

    dvs = Array(Float64, nb_cars)
    dys = Array(Float64, nb_cars)

    # agent
    dvs[1] = a.acc*dt
    dys[1] = a.lane_change

    if a.acc < -mdp.rmodel.dangerous_brake_threshold
        r -= mdp.rmodel.cost_emergency_brake
    end
    if s.env_cars[1].pos[y] == mdp.rmodel.desired_lane
        r += mdp.rmodel.reward_in_desired_lane
    end

    changers = IntSet()
    for i in 2:nb_cars
        neighborhood = get_neighborhood(mdp.dmodel, s, i)

        # To distinguish between different models--is there a better way?
        behavior = get(s.env_cars[i].behavior)

        acc = generate_accel(behavior, mdp.dmodel, neighborhood, s, i, rng)
        dvs[i] = dt*acc
        if acc < -mdp.rmodel.dangerous_brake_threshold
            r -= mdp.rmodel.cost_emergency_brake
        end

        sp.env_cars[i].lane_change = generate_lane_change(behavior, mdp.dmodel, neighborhood, s, i, rng)
        dys[i] = sp.env_cars[i].lane_change * dmodel.lane_change_vel * dt
        if sp.env_cars[i].lane_change
            push!(changers, i)
        end
    end

    ## Consistency checking ##
    #========================#

    sorted_changers = sort!(collect(changers), by=i->s.env_cars[i].pos[1]) # this might be slow because anonymous functions are slow

    # iterate through pairs
    iter_state = start(sorted_changers)
    j, iter_state = next(sorted_changers, iter_state)
    while !done(sorted_changers, state)
        i = j
        j, iter_state = next(sorted_changers, iter_state)
        car_i = s.env_cars[i]
        car_j = s.env_cars[j]

        # check if they are both starting to change lanes on this step
        if isinteger(car_i.pos[2]) && isinteger(car_j.pos[2])

            # make sure there is a conflict longitudinally
            if car_i.pos[1] - car_j.pos[1] <= pp.l_car

                # check if they are near each other lanewise
                if abs(car_i.pos[2] - car_j.pos[2]) <= 2.0

                    # check if they are moving towards each other
                    if dys[i]*dys[j] < 0.0 && abs(car_i.pos[2]+dys[2] - car_j.pos[2]+dys[2]) < 2.0

                        # make j stay in his lane
                        dys[j] = 0.0
                        car_states_[j].lane_change = 0.0
                    end
                end
            end
        end
    end

    ## Dynamics and Exits ##
    #======================#

    exits = IntSet()
    for i in 1:nb_cars
        sp.env_cars[i].pos[1] = s.env_cars[i].pos[1] + dt*(s.env_cars[i].vel - s.env_cars[1].vel)
        sp.env_cars[i].pos[2] = s.env_cars[i].pos[2] + dys[i]
        sp.env_cars[i].vel = max(min(s.env_cars[i].vel + dvs[i],pp.v_max),pp.v_min)

        # note lane change is updated above
        if sp.env_cars[i].pos[1] < 0.0 || sp.env_cars[i].pos[1] >= lane_length
            push!(exits, i)
        end
    end
    deleteat!(sp.env_cars, exits)
    nbcars -= length(exits)

    ## Generate new cars ##
    #=====================#

    for j in 1:(pp.nb_env_cars-nb_cars)
        if rand(rng) <= mdp.dmodel.p_appear
            # calculate clearance for all the lanes
            clearances = Dict{Tuple{Int,Bool},Float64}() # integer is lane, bool is true if front, false if back
            for i in 1:pp.nb_lanes, j in (true,false)
                clearances[(i,j)] = Inf
            end
            for i in nb_cars
                lowlane = floor(Int, s.env_cars[i].pos[2])
                highlane = ceil(Int, s.env_cars[i].pos[2])
                front = pp.lane_length - (s.env_cars[i].pos[1] + pp.l_car) # l_car is half the length of the old car plus half the length of the new one
                back = s.env_cars[i].pos[1] - pp.l_car
                clearances[(lowlane, true)] = min(front, clearances[(lowlane, true)])
                clearances[(highlane, true)] = min(front, clearances[(highlane, true)])
                clearances[(lowlane, false)] = min(back, clearances[(lowlane, false)])
                clearances[(highlane, false)] = min(back, clearances[(highlane, false)])
            end
            clear_spots = Array(Tuple{Int,Bool}, 0)
            for i in 1:pp.nb_lanes, j in (true,false)
                if clearances[(i,j)] >= mdp.dmodel.appear_clearance
                    push!(clear_spots, (i,j))
                end
            end
            # pick one
            spot = rand(rng, clear_spots)
            behavior = sample(mdp.dmodel.behaviors, mdp.dmodel.behavior_probabilities)
            if spot[2] # at front
                push!(sp.env_cars, CarState((pp.lane_length, spot[1]), sp.env_car[1], pp.lane_length, behavior))
            else # at back
                push!(sp.env_cars, CarState((pp.lane_length, spot[1]), sp.env_car[1], 0.0, behavior))
            end
        end
    end

    return (sp, r)
end

function initial_state(mdp::NoCrashMDP, rng::AbstractRNG, s::MLState=create_state(mdp))

  pp = mdp.dmodel.phys_param

  #Unif # cars in initial scene
  _nb_cars = rand(rng,1:mdp.dmodel.nb_cars)

  #place ego car
  s.env_cars[1].pos[1] = pp.lane_length/2. #this is fixed
  s.env_cars[1].pos[2] = rand(rng,1:(pp.nb_lanes*2-1))
  #ego velocity
  s.env_cars[1].vel = max(min(randn(rng)*mdp.dmodel.vel_sigma + pp.v_med, pp.v_max), pp.v_min)

  # XXX dirichlet and exponential are from distributions--does not accept rng!!!
  dir_distr = Dirichlet(mdp.dmodel.lane_weights)
  cars_per_lane = floor(Int,_nb_cars*rand(dir_distr))

  # TODO remove nb_cars - sum(cars_per_lane)
  dist_distr1 = Exponential(mdp.dmodel.dist_lambda)

  idx = 2
  for (_lane,nb_cars) in enumerate(cars_per_lane)

    if nb_cars <= 0
      continue
    end

    # if there's no room to generate the remaining cars
    break_flag = false

    lane = 2*_lane - 1
    #from front to back
    for i = 1:nb_cars
      if break_flag
        continue
      end
      #sample Behavior TODO sample(rng,v,wv) in utils?
      behavior = sample(mdp.dmodel.behaviors,
                        mdp.dmodel.behavior_probabilities)
      #sample velocity
      #TODO need generic interface with behavior models for desired speed
      vel = randn(rng)*mdp.dmodel.vel_sigma + behavior.p_idm.v0
      vel = max(min(vel,pp.v_max),pp.v_min)
      if i == 1
        dist = rand(dist_distr1)
        x = pp.lane_length - dist
        if x < 0.
          break_flag = true
          continue
        end
        s.env_cars[idx].pos = (x,lane,)
      else
        #mean of v0*T - min_dist
        lam = 1/(behavior.p_idm.T*behavior.p_idm.v0 - mdp.dmodel.appear_clearance)
        assert(lam > 0.)
        dist_distr = Exponential(lam)
        dist = rand(dist_distr) + mdp.dmodel.appear_clearance
        x = s.env_cars[idx-1].pos[1]-dist
        if x < 0.
          break_flag = true
          continue
        end
        s.env_cars[idx].pos = (x,lane,)
      end

      s.env_cars[idx].vel = vel
      #TODO lanechanging? initialized as zero
      s.env_cars[idx].behavior = Nullable{IDMMOBILBehavior}(behavior)
      idx += 1
    end

  end

  return s

end
