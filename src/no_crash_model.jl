type NoCrashRewardModel <: AbstractMLRewardModel
    cost_emergency_brake::Float64
    reward_in_desired_lane::Float64

    dangerous_brake_threshold::Float64 # if the deceleration is greater than this cost_emergency_brake will be accured
    desired_lane::Int
end
#XXX temporary
NoCrashRewardModel() = NoCrashRewardModel(-10.,100.,-3.,1)

type NoCrashIDMMOBILModel <: AbstractMLDynamicsModel
    nb_cars::Int
    phys_param::PhysicalParam

    behaviors::Vector{BehaviorModel}
    behavior_probabilities::WeightVec

    adjustment_acceleration::Float64
    lane_change_vel::Float64

    p_appear::Float64 # probability of a new car appearing if the maximum number are not on the road
    appear_clearance::Float64 # minimum clearance for a car to appear

    vel_sigma::Float64 # std of new car speed about v0
    lane_weights::Array{Float64,1} # dirichlet alpha values for each lane: first is for rightmost lane
    dist_var::Float64 # variance of distance--can back out rate, shape param from this

    #XXX temporary
    function NoCrashIDMMOBILModel(nb_cars::Int,pp::PhysicalParam)
      self = new()
      self.nb_cars = nb_cars
      self.phys_param = pp

      # XXX VVV TEMP XXX
      self.behaviors = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in
                                          enumerate(product(["cautious","normal","aggressive"],
                                          [pp.v_slow+0.5;pp.v_med;pp.v_fast],
                                          [pp.l_car]))]
      self.behavior_probabilities = WeightVec(ones(length(self.behaviors)))
      self.adjustment_acceleration = 1. #XXX
      self.lane_change_vel = 1. #XXX
      self.p_appear = 0.5
      self.appear_clearance = 4.
      self.vel_sigma = 2.
      self.lane_weights = ones(pp.nb_lanes)
      self.dist_var = 2 #idk

      return self
    end
end

typealias NoCrashMDP MLMDP{MLState, MLAction, NoCrashIDMMOBILModel, NoCrashRewardModel}

# action space = {a in {accelerate,maintain,decelerate}x{left_lane_change,maintain,right_lane_change} | a is safe} U {EMERGENCY_BRAKE}
immutable NoCrashActionSpace
    NORMAL_ACTIONS::Vector{MLAction} # all the actions except emergency brake
    acceptable::IntSet
    emergency_brake::MLAction
end
# TODO for performance, make this a macro?
const NB_NORMAL_ACTIONS = 9

function NoCrashActionSpace(mdp::NoCrashMDP)
    accels = (-mdp.adjustment_acceleration, 0.0, mdp.adjustment_acceleration)
    lane_changes = (-1, 0, 1)
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
    if length(s.env_cars) < 2
        return 0.0
    end
    dt = mdp.dmodel.phys_param.dt
    v_min = mdp.dmodel.phys_param.v_min
    l_car = mdp.dmodel.phys_param.l_car
    ego = s.env_cars[1]

    car_in_front = 0
    smallest_gap = Inf
    # find car immediately in front
    if length(s.env_cars > 1)
        for i in 2:nb_cars
            if occupation_overlap(s.env_cars[i].y, ego.y) # occupying same lane
                gap = s.env_cars[i].x - ego.x
                if gap >= 0.0 && gap < smallest_gap
                    car_in_front = i
                    smallest_gap = gap
                end
            end
        end
        # calculate necessary acceleration
        if car_in_front == 0
            return 0.0
        else
            other = s.env_cars[car_in_front]
            return (gap - l_car + other.vel*dt + v_min*dt - 2.0*ego.vel*dt) / dt^2
        end
    end
end

"""
Returns true if cars at y1 and y1 occupy the same lane
"""
function occupation_overlap(y1::Float64, y2::Float64)
    return abs(y1-y2) < 1.0 || ceil(Int, y1) == floor(Int, y2) || floor(Int, y1) == ceil(Int, y2)
end

#XXX temp
create_state(p::NoCrashMDP) = MLState(false, 1, p.dmodel.phys_param.v_med, CarState[CarState(-1.,1,1.,0,p.dmodel.behaviors[1]) for _ = 1:p.dmodel.nb_cars])

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
    if s.env_cars[1].y == mdp.rmodel.desired_lane
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

    sorted_changers = sort!(collect(changers), by=i->s.env_cars[i].x) # this might be slow because anonymous functions are slow

    # iterate through pairs
    iter_state = start(sorted_changers)
    j, iter_state = next(sorted_changers, iter_state)
    while !done(sorted_changers, state)
        i = j
        j, iter_state = next(sorted_changers, iter_state)
        car_i = s.env_cars[i]
        car_j = s.env_cars[j]

        # check if they are both starting to change lanes on this step
        if isinteger(car_i.y) && isinteger(car_j.y)

            # make sure there is a conflict longitudinally
            if car_i.x - car_j.x <= pp.l_car

                # check if they are near each other lanewise
                if abs(car_i.y - car_j.y) <= 2.0

                    # check if they are moving towards each other
                    if dys[i]*dys[j] < 0.0 && abs(car_i.y+dys[2] - car_j.y+dys[2]) < 2.0

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
        sp.env_cars[i].x = s.env_cars[i].x + dt*(s.env_cars[i].vel - s.env_cars[1].vel)
        sp.env_cars[i].y = s.env_cars[i].y + dys[i]
        sp.env_cars[i].vel = max(min(s.env_cars[i].vel + dvs[i],pp.v_max),pp.v_min)

        # note lane change is updated above
        if sp.env_cars[i].x < 0.0 || sp.env_cars[i].x >= lane_length
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
                lowlane = floor(Int, s.env_cars[i].y)
                highlane = ceil(Int, s.env_cars[i].y)
                front = pp.lane_length - (s.env_cars[i].x + pp.l_car) # l_car is half the length of the old car plus half the length of the new one
                back = s.env_cars[i].x - pp.l_car
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
            behavior = sample(rng, mdp.dmodel.behaviors, mdp.dmodel.behavior_probabilities)
            if spot[2] # at front
                push!(sp.env_cars, CarState(pp.lane_length, spot[1], sp.env_car[1], pp.lane_length, behavior))
            else # at back
                push!(sp.env_cars, CarState(pp.lane_length, spot[1], sp.env_car[1], 0.0, behavior))
            end
        end
    end

    sp.crashed = is_crash(mdp, s, a)

    return (sp, r)
end

function initial_state(mdp::NoCrashMDP, rng::AbstractRNG, s::MLState=create_state(mdp))

  pp = mdp.dmodel.phys_param
  #Unif # cars in initial scene
  _nb_cars = rand(rng,1:mdp.dmodel.nb_cars)
  #place ego car

  pos_x = pp.lane_length/2. #this is fixed
  pos_y = rand(rng,1:(pp.nb_lanes*2-1))
  #ego velocity
  vel = max(min(randn(rng)*mdp.dmodel.vel_sigma + pp.v_med, pp.v_max), pp.v_min)

  s.env_cars[1] = CarState(pos_x,pos_y,vel,0,Nullable{BehaviorModel}())
  # XXX dirichlet and exponential are from distributions--does not accept rng!!!
  dir_distr = Dirichlet(mdp.dmodel.lane_weights)
  cars_per_lane = floor(Int,_nb_cars*rand(dir_distr))

  # TODO remove nb_cars - sum(cars_per_lane)
  dist_distr1 = Exponential(1./mdp.dmodel.dist_var)

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
      behavior = sample(rng, mdp.dmodel.behaviors,
                        mdp.dmodel.behavior_probabilities)

      #TODO check where the ego car is!!!
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
      else
        #mean of v0*T - min_dist
        mu = behavior.p_idm.T*behavior.p_idm.v0 - mdp.dmodel.appear_clearance
        var = mdp.dmodel.dist_var
        assert(mu > 0.)
        dist_distr = Gamma((mu^2)/(var^2),(var^2)/mu)
        dist = rand(dist_distr) + mdp.dmodel.appear_clearance + mdp.dmodel.phys_param.l_car
        x = s.env_cars[idx-1].x-dist
        #in ego lane
        if lane == (pos_y+1)/2.
          #do something idk
        end
        if x < 0.
          break_flag = true
          continue
        end
      end
      car = CarState(x,lane,vel,0,behavior)
      s.env_cars[idx] = car
      idx += 1
    end

  end

  return s

end
