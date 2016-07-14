type NoCrashRewardModel <: AbstractMLRewardModel
    cost_dangerous_brake::Float64 # POSITIVE NUMBER
    reward_in_desired_lane::Float64 # POSITIVE NUMBER

    dangerous_brake_threshold::Float64 # (POSITIVE NUMBER) if the deceleration is greater than this cost_dangerous_brake will be accured
    desired_lane::Int
end
#XXX temporary
NoCrashRewardModel() = NoCrashRewardModel(100.,10.,8.0,4)

type NoCrashIDMMOBILModel <: AbstractMLDynamicsModel
    nb_cars::Int
    phys_param::PhysicalParam

    behaviors::Vector{BehaviorModel}
    behavior_probabilities::WeightVec

    adjustment_acceleration::Float64
    lane_change_rate::Float64 # in LANES PER SECOND

    p_appear::Float64 # probability of a new car appearing if the maximum number are not on the road
    appear_clearance::Float64 # minimum clearance for a car to appear

    vel_sigma::Float64 # std of new car speed about v0
    lane_weights::Array{Float64,1} # dirichlet alpha values for each lane: first is for rightmost lane
    dist_var::Float64 # variance of distance--can back out rate, shape param from this

    lane_terminate::Bool # if true, terminate the simulation when the car has reached the desired lane
end

#XXX temporary
function NoCrashIDMMOBILModel(nb_cars::Int, pp::PhysicalParam; lane_terminate=false)
    behaviors=IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in
                       enumerate(product(["cautious","normal","aggressive"],
                              [pp.v_slow+0.5;pp.v_med;pp.v_fast],
                              [pp.l_car]))]
    return NoCrashIDMMOBILModel(
        nb_cars,
        pp,
        behaviors,
        WeightVec(ones(length(behaviors))),
        1.,
        1.0/(2.0*pp.dt), # lane change rate
        0.5,
        20.0,
        0.5,
        ones(pp.nb_lanes),
        10.^2,
        lane_terminate
    )
end

typealias NoCrashMDP MLMDP{MLState, MLAction, NoCrashIDMMOBILModel, NoCrashRewardModel}

typealias NoCrashPOMDP MLPOMDP{MLState, MLAction, MLObs, NoCrashIDMMOBILModel, NoCrashRewardModel}

# TODO issue here VVV need a different way to create observation
create_action(::Union{NoCrashMDP,NoCrashPOMDP}) = MLAction()

# action space = {a in {accelerate,maintain,decelerate}x{left_lane_change,maintain,right_lane_change} | a is safe} U {brake}
immutable NoCrashActionSpace <: AbstractSpace{MLAction}
    NORMAL_ACTIONS::Vector{MLAction} # all the actions except brake
    acceptable::IntSet
    brake::MLAction # this action will be EITHER braking at half the dangerous brake threshold OR the braking necessary to prevent a collision at all time in the future
end
# TODO for performance, make this a macro?
const NB_NORMAL_ACTIONS = 9

function NoCrashActionSpace(mdp::Union{NoCrashMDP,NoCrashPOMDP})
    accels = (-mdp.dmodel.adjustment_acceleration, 0.0, mdp.dmodel.adjustment_acceleration)
    lane_changes = (-mdp.dmodel.lane_change_rate, 0.0, mdp.dmodel.lane_change_rate)
    NORMAL_ACTIONS = MLAction[MLAction(a,l) for (a,l) in product(accels, lane_changes)]
    return NoCrashActionSpace(NORMAL_ACTIONS, IntSet(), MLAction()) # note: brake will be calculated later based on the state
end

function actions(mdp::Union{NoCrashMDP,NoCrashPOMDP})
    return NoCrashActionSpace(mdp)
end

function actions(mdp::Union{NoCrashMDP,NoCrashPOMDP}, s::Union{MLState, MLPhysicalState}, as::NoCrashActionSpace) # no implementation without the third arg to enforce efficiency
    acceptable = IntSet()
    for i in 1:NB_NORMAL_ACTIONS
        a = as.NORMAL_ACTIONS[i]
        ego_y = s.env_cars[1].y
        # prevent going off the road
        if ego_y == 1. && a.lane_change < 0. || ego_y == mdp.dmodel.phys_param.nb_lanes && a.lane_change > 0.0
            continue
        end
        # prevent running into the person in front or to the side
        if is_safe(mdp, s, as.NORMAL_ACTIONS[i])
            push!(acceptable, i)
        end
    end
    brake_acc = min(max_safe_acc(mdp,s), -mdp.rmodel.dangerous_brake_threshold/2.0)
    brake = MLAction(brake_acc, 0)
    return NoCrashActionSpace(as.NORMAL_ACTIONS, acceptable, brake)
end

iterator(as::NoCrashActionSpace) = as
Base.start(as::NoCrashActionSpace) = 1
function Base.next(as::NoCrashActionSpace, state::Integer)
    while !(state in as.acceptable)
        if state > NB_NORMAL_ACTIONS
            return (as.brake, state+1)
        end
        state += 1
    end
    return (as.NORMAL_ACTIONS[state], state+1)
end
Base.done(as::NoCrashActionSpace, state::Integer) = state > NB_NORMAL_ACTIONS+1
Base.length(as::NoCrashActionSpace) = length(as.acceptable) + 1

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
Calculate the maximum safe acceleration that will allow the car to avoid a collision if the car in front slams on its brakes
"""
function max_safe_acc(mdp::Union{NoCrashMDP,NoCrashPOMDP}, s::Union{MLState,MLObs}, lane_change::Float64=0.0)
    dt = mdp.dmodel.phys_param.dt
    v_min = mdp.dmodel.phys_param.v_min
    l_car = mdp.dmodel.phys_param.l_car
    bp = mdp.dmodel.phys_param.brake_limit
    ego = s.env_cars[1]

    car_in_front = 0
    smallest_gap = Inf
    # find car immediately in front
    if length(s.env_cars) > 1
        for i in 2:length(s.env_cars)#nb_cars
            if occupation_overlap(s.env_cars[i].y, 0.0, ego.y, lane_change) # occupying same lane
                gap = s.env_cars[i].x - ego.x - l_car
                if gap >= -l_car && gap < smallest_gap
                    car_in_front = i
                    smallest_gap = gap
                end
            end
        end
        # calculate necessary acceleration
        if car_in_front == 0
            return Inf
        else
            return max_safe_acc(smallest_gap, ego.vel, s.env_cars[car_in_front].vel, bp, dt)
        end
    end
    return Inf
end

"""
Return the maximum acceleration that the car behind can have on this step so that it won't hit the car in front if it slams on its brakes
"""
function max_safe_acc(gap, v_behind, v_ahead, braking_limit, dt)
    bp = braking_limit
    v = v_behind
    vo = v_ahead
    g = gap
    bp = braking_limit
    # VVV see mathematica notebook
    return - (bp*dt + 2.*v - sqrt(8.*g*bp + bp^2*dt^2 - 4.*bp*dt*v + 4.*vo^2)) / (2.*dt)
end

#=
# I think this is wrong (7/13)
"""
Calculate the maximum distance that the car could achieve if it uses the maximum acceleration

The first argument is a behavior model so that this can be dispatched based on the behavior model
"""
function max_dx(b::IDMMOBILBehavior, cs::CarState, dt)
    return cs.x + (cs.vel + b.p_idm.a/2.0)*dt
end

max_dx(cs::CarStateObs, dt::Float64) = cs.x + (cs.vel + 2.1/2.)*dt # Max accel is 2.0 in aggressive
=#

# """
# Test whether, if the ego vehicle takes action a, it will always be able to slow down fast enough if the car in front slams on his brakes and won't pull in front of another car so close they can't stop
# """

function is_safe(mdp::Union{NoCrashMDP,NoCrashPOMDP}, s::Union{MLState,MLObs}, a::MLAction)
    if a.acc >= max_safe_acc(mdp, s, a.lane_change)
        return false
    end
    # check whether we will go into anyone else's lane so close that they might hit us or we might run into them
    if isinteger(s.env_cars[1].y) && a.lane_change != 0.0
        dt = mdp.dmodel.phys_param.dt
        l_car = mdp.dmodel.phys_param.l_car
        for i in 2:length(s.env_cars)
            car = s.env_cars[i]
            ego = s.env_cars[1]
            if car.x < ego.x + l_car && occupation_overlap(ego.y, a.lane_change, car.y, 0.0)  # ego is in front of car
                # #   Note: can overestimate max_dx (operate conservatively)
                # #  this doesn't seem right - this only checks one step into the future
                # dx = typeof(s)<:MLState ? max_dx(get(car.behavior), car, dt) : max_dx(car, dt)
                # if ego.x + (ego.vel + dt*a.acc/2.0)*dt - (car.x + dx) < mdp.dmodel.phys_param.l_car
                #     return false
                # end
                
                # New definition of safe - the car behind can brake at max braking to avoid the ego if the ego
                # slams on his brakes
                # XXX IS THIS RIGHT??
                # I think I need a better definition of "safe" here
                gap = ego.x - car.x - l_car
                if gap <= 0.0
                    return false
                end
                braking_acc = max_safe_acc(gap, car.vel, ego.vel, mdp.dmodel.phys_param.brake_limit, dt)
                # if braking_acc < -mdp.dmodel.phys_param.brake_limit
                if braking_acc < 0.0
                    return false
                end
            end
        end
    end
    return true
end

#XXX temp
create_state(p::Union{NoCrashMDP,NoCrashPOMDP}) = MLState(false, Array(CarState, p.dmodel.nb_cars))
create_observation(pomdp::NoCrashPOMDP) = MLObs(false, Array(CarStateObs, pomdp.dmodel.nb_cars))


function generate_s(mdp::Union{NoCrashMDP,NoCrashPOMDP}, s::MLState, a::MLAction, rng::AbstractRNG, sp::MLState=create_state(mdp))

    pp = mdp.dmodel.phys_param
    dt = pp.dt
    nb_cars = length(s.env_cars)
    resize!(sp.env_cars, nb_cars)

    ## Calculate deltas ##
    #====================#

    dxs = Array(Float64, nb_cars)
    dvs = Array(Float64, nb_cars)
    dys = Array(Float64, nb_cars)
    lcs = Array(Float64, nb_cars)

    # agent
    dvs[1] = a.acc*dt
    dxs[1] = s.env_cars[1].vel*dt + a.acc*dt^2/2.
    lcs[1] = a.lane_change
    dys[1] = a.lane_change*dt

    changers = IntSet()
    for i in 2:nb_cars
        neighborhood = get_neighborhood(pp, s, i)

        # To distinguish between different models--is there a better way?
        behavior = get(s.env_cars[i].behavior)

        acc = generate_accel(behavior, mdp.dmodel, s, neighborhood, i, rng)
        dvs[i] = dt*acc
        dxs[i] = (s.env_cars[i].vel + dvs[i]/2.)*dt

        lcs[i] = generate_lane_change(behavior, mdp.dmodel, s, neighborhood, i, rng)
        dys[i] = lcs[i] * dt
        if lcs[i] != 0
            push!(changers, i)
        end
    end

    ## Consistency checking ##
    #========================#

    sorted_changers = sort!(collect(changers), by=i->s.env_cars[i].x, rev=true) # this might be slow because anonymous functions are slow

    if length(sorted_changers) >= 2 #something to compare
      # iterate through pairs
      iter_state = start(sorted_changers)
      j, iter_state = next(sorted_changers, iter_state)
      while !done(sorted_changers, iter_state)
          i = j
          j, iter_state = next(sorted_changers, iter_state)
          car_i = s.env_cars[i]
          car_j = s.env_cars[j]

          # check if they are both starting to change lanes on this step
          if isinteger(car_i.y) && isinteger(car_j.y)

              # make sure there is a conflict longitudinally
              if car_i.x - car_j.x <= pp.l_car || car_i.x + dxs[i] - car_j.x + dxs[j] <= pp.l_car

                  # check if they are near each other lanewise
                  if abs(car_i.y - car_j.y) <= 2.0

                      # check if they are moving towards each other
                      if dys[i]*dys[j] < 0.0 && abs(car_i.y+dys[i] - car_j.y+dys[j]) < 2.0

                          # make j stay in his lane
                          dys[j] = 0.0
                          lcs[j] = 0.0
                      end
                  end
              end
          end
      end
    end

    ## Dynamics and Exits ##
    #======================#

    exits = IntSet()
    for i in 1:nb_cars
        car = s.env_cars[i]
        xp = car.x + (dxs[i] - dxs[1])
        yp = car.y + dys[i]
        velp = max(min(car.vel + dvs[i],pp.v_max), pp.v_min)
        # note lane change is updated above

        # check if a lane was crossed and snap back to it
        if isinteger(car.y)
            # prevent a multi-lane change in a single timestep
            if abs(yp-car.y) > 1.
                yp = car.y + sign(dys[i])
            end
        else # car.y is not an integer
            if floor(yp) >= ceil(car.y)
                yp = ceil(car.y)
            end
            if ceil(yp) <= floor(car.y)
                yp = floor(car.y)
            end
        end

        # enforce max/min y position constraints
        # yp = min(max(yp,1.), pp.nb_lanes) # this should never be violated because of the check above
        @assert yp >= 1.0 && yp <= pp.nb_lanes

        if xp < 0.0 || xp >= pp.lane_length
            push!(exits, i)
        else
            sp.env_cars[i] = CarState(xp, yp, velp, lcs[i], car.behavior, s.env_cars[i].id)
        end
    end
    deleteat!(sp.env_cars, exits)
    nb_cars -= length(exits)

    ## Generate new cars ##
    #=====================#

    if nb_cars < mdp.dmodel.nb_cars && rand(rng) <= mdp.dmodel.p_appear
        # calculate clearance for all the lanes
        clearances = Dict{Tuple{Int,Bool},Float64}() # integer is lane, bool is true if front, false if back
        for i in 1:pp.nb_lanes, j in (true,false)
            clearances[(i,j)] = Inf
        end
        for i in 1:nb_cars
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

        if length(clear_spots) > 0
            # pick one
            spot = rand(rng, clear_spots)

            next_id = maximum([c.id for c in s.env_cars]) + 1
            behavior = sample(rng, mdp.dmodel.behaviors, mdp.dmodel.behavior_probabilities)
            if spot[2] # at front
                velp = sp.env_cars[1].vel - rand(rng) * min(mdp.dmodel.vel_sigma, sp.env_cars[1].vel - pp.v_min)
                push!(sp.env_cars, CarState(pp.lane_length, spot[1], velp, 0.0, behavior, next_id))
            else # at back
                velp = rand(rng) * min(mdp.dmodel.vel_sigma, pp.v_max - sp.env_cars[1].vel) + sp.env_cars[1].vel
                push!(sp.env_cars, CarState(0.0, spot[1], velp, 0.0, behavior, next_id))
            end
        end
    end

    # sp.crashed = is_crash(mdp, s, sp, warning=false)
    sp.crashed = false

    @assert sp.env_cars[1].x == s.env_cars[1].x # ego should not move

    return sp
end

function reward(mdp::Union{NoCrashMDP, NoCrashPOMDP}, s::MLState, ::MLAction, sp::MLState)
    r = 0.0
    if sp.env_cars[1].y == mdp.rmodel.desired_lane
        r += mdp.rmodel.reward_in_desired_lane
    end
    nb_brakes = detect_braking(mdp, s, sp)
    r -= mdp.rmodel.cost_dangerous_brake*nb_brakes
    return r
end

"""
Return the number of braking actions that occured during this state transition
"""
function detect_braking(mdp::Union{NoCrashMDP,NoCrashPOMDP}, s::MLState, sp::MLState, threshold=mdp.rmodel.dangerous_brake_threshold)
    nb_brakes = 0
    nb_leaving = 0 
    dt = mdp.dmodel.phys_param.dt
    for (i,c) in enumerate(s.env_cars)
        if length(sp.env_cars) >= i-nb_leaving
            cp = sp.env_cars[i-nb_leaving]
        else
            break
        end
        if cp.id != c.id
            nb_leaving += 1
            continue
        else
            acc = (cp.vel-c.vel)/dt
            if acc < -threshold
                nb_brakes += 1
            end
        end
    end
    # @assert nb_leaving <= 5 # sanity check - can remove this if it is violated as long as it doesn't happen all the time
    return nb_brakes
end

function initial_state(mdp::Union{NoCrashMDP,NoCrashPOMDP}, rng::AbstractRNG, s::MLState=create_state(mdp))

  srand(rand(rng, UInt32))
  pp = mdp.dmodel.phys_param
  #Unif # cars in initial scene
  #_nb_cars = rand(rng,floor(Int, mdp.dmodel.nb_cars/2):mdp.dmodel.nb_cars)
  _nb_cars = mdp.dmodel.nb_cars
  #place ego car

  pos_x = pp.lane_length/2. #this is fixed
  pos_y = 1. #rand(rng,1:(pp.nb_lanes)) # NOTE wlog start in lane 1, goal is leftmost lane
  #ego velocity
  vel = max(min(randn(rng)*mdp.dmodel.vel_sigma + pp.v_med, pp.v_max), pp.v_min)

  s.env_cars[1] = CarState(pos_x, pos_y, vel, 0, Nullable{BehaviorModel}(), 1)
  # XXX dirichlet and exponential are from distributions--does not accept rng!!!
  dir_distr = Dirichlet(mdp.dmodel.lane_weights)
  cars_per_lane = floor(Int,_nb_cars*rand(dir_distr))

  dist_distr1 = Exponential(sqrt(mdp.dmodel.dist_var))
  # dist_distr1 = Exponential(1.0/mdp.dmodel.dist_var)

  last_front = 1 #last car to be sampled in front of ego car--starts as ego car
  last_back = 1 #last car to be sampled behind ego car -- starts with ego car


  idx = 2
  for (lane,nb_cars) in enumerate(cars_per_lane)

    if nb_cars == 0
      continue
    end

    # if there's no room to generate the remaining cars
    break_flag = false

    #lane = 2*_lane - 1
    #from front to back

    for i = 1:nb_cars
      if break_flag
        continue
      end
      #sample Behavior TODO sample(rng,v,wv) in utils?
      behavior = sample(rng, mdp.dmodel.behaviors,
                        mdp.dmodel.behavior_probabilities)
      #TODO need generic interface with behavior models for desired speed

      if lane == pos_y
        #ego car in this lane: alternate sampling
        sample_front = rand(rng,Bool)
        if sample_front
          dist = sample_distance(mdp.dmodel,
                                  get(s.env_cars[last_front].behavior,
                                      mdp.dmodel.behaviors[1]), #XXX temp
                                  rng)
          x = s.env_cars[last_front].x + dist
          last_front = idx
        else #sample back
          dist = sample_distance(mdp.dmodel,behavior,rng)
          x = s.env_cars[last_back].x - dist
          last_back = idx
        end
      else
        #ego car not in this lane: sample from front to back
        if i == 1
          dist = rand(dist_distr1)
          x = pp.lane_length - dist
        else
          # mean of v0*T - min_dist
          dist = sample_distance(mdp.dmodel,behavior,rng)
          x = s.env_cars[idx-1].x-dist
        end
      end
      if x < 0. || x > mdp.dmodel.phys_param.lane_length
        break_flag = true
        continue
      end
      #sample velocity
      vel = randn(rng)*mdp.dmodel.vel_sigma + behavior.p_idm.v0
      vel = max(min(vel,pp.v_max),pp.v_min)

      car = CarState(x, lane, vel, 0, behavior, idx)
      s.env_cars[idx] = car
      idx += 1
    end

  end

  resize!(s.env_cars,idx-1)

  return s

end

function sample_distance(dmodel::NoCrashIDMMOBILModel, behavior::IDMMOBILBehavior, rng::AbstractRNG)
  mu = behavior.p_idm.T*behavior.p_idm.v0 - dmodel.appear_clearance
  var = dmodel.dist_var
  assert(mu > 0.)
  dist_distr = Gamma((mu^2)/var,var/mu)
  dist = rand(dist_distr) + dmodel.appear_clearance + dmodel.phys_param.l_car

  return dist
end


function generate_o(mdp::Union{NoCrashMDP,NoCrashPOMDP}, s::MLState, a::MLAction, sp::MLState, o::MLObs=create_observation(mdp))
  #TODO add noise? no?

  return MLObs(sp)

end

# TODO generate_sor

function generate_sor(pomdp::NoCrashPOMDP, s::MLState, a::MLAction, rng::AbstractRNG, sp::MLState, o::MLObs)
    sp, r = generate_sr(pomdp, s, a, rng, sp)
    o = generate_o(pomdp, s, a, sp, o)
    return sp, o, r
end

function pdf(mdp::Union{NoCrashMDP,NoCrashPOMDP}, sp::MLState, a::MLAction, o::MLObs)

  """"
  P(o|s', a) (unweighted)
  The degree of similarity between states?
  say similarity ~ 1/distance between states

  Ignore ego car: presumably will have mostly deterministic dynamics?
  """

  dist = 0.

  id = Dict{Int,Int}([car.id=>i+1 for (i,car) in enumerate(s.env_cars[2:end])])
	idp = Dict{Int,Int}([car.id=>i+1 for (i,car) in enumerate(sp.env_cars[2:end])])

  """
  Alternatively: assume x,y,v,lc are correct, uncertainty about behavior model
  """


end

discount(mdp::Union{MLMDP,MLPOMDP}) = mdp.discount
isterminal(mdp::Union{MLMDP,MLPOMDP},s::MLState) = s.crashed

function isterminal(mdp::Union{NoCrashMDP,NoCrashPOMDP}, s::MLState)
    if s.crashed
        return true
    elseif mdp.dmodel.lane_terminate && s.env_cars[1].y == mdp.rmodel.desired_lane
        return true
        println("lane termination")
    else
        return false
    end
end
