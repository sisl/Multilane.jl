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
            if occupation_overlap(s.env_cars[i].pos[2], ego.pos[2]) # occupying same lane
                gap = s.env_cars[i].pos[1] - ego.pos[1]
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

function generate_sr(mdp::NoCrashMDP, s::MLState, a::MLAction, rng::AbstractRNG, sp::MLState=create_state(mdp))

    pp = mdp.dmodel.phys_param
    dt = pp.dt
    nb_cars = length(s.env_cars)
    resize!(sp.env_cars, nb_cars)
    r = 0.0 # reward

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

    sp.crashed = is_crash(mdp, s, a)

    return (sp, r)
end

function initial_state(mdp::NoCrashMDP, rng::AbstractRNG, s::MLState=create_state(mdp))
  #TODO how to make a state that's not instant death?
  
end
