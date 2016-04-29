type NoCrashRewardModel <: AbstractMLRewardModel
    cost_emergency_brake::Float64
    reward_in_desired_lane::Float64

    dangerous_brake_threshold::Float64 # if the deceleration is greater than this cost_emergency_brake will be accured
end

type NoCrashIDMMOBILModel <: AbstractMLDynamicsModel
    nb_cars::Int
    phys_param::PhysicalParam

    behaviors::Vector{(Symbol, BehaviorModel)}
    behavior_probabilities::WeightVector

    adjustment_acceleration::Float64
end

typealias NoCrashMPD MLMDP{MLState, MLAction, NoCrashIDMMOBILModel, NoCrashRewardModel}

# action space = {a in {accelerate,maintain,decelerate}x{left_lane_change,maintain,right_lane_change} | a is safe} U {EMERGENCY_BRAKE}
immutable NoCrashActionSpace
    NORMAL_ACTIONS::Vector{MLAction} # all the actions except emergency brake
    acceptable::IntSet
    emergency_brake::MLAction
end
# TODO for performance, make this a macro?
const NB_NORMAL_ACTIONS::Int = 9

function NoCrashActionSpace(mdp::NoCrashMPD)
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

function generate_sr(mdp::NoCrashMPD, s::MLState, a::MLAction, rng::AbstractRNG)

    pp = mdp.dmodel.phys_param
    dt = pp.dt
    nb_cars = length(s.env_cars)
    car_states_ = resize!(sp.env_cars,0)

    agent_lane_ = s.agent_pos + a.lane_change
    agent_lane_ = max(1,min(agent_lane_,nb_col)) #can't leave the grid

    agent_vel_ = s.agent_vel + a.acc*dt
    agent_vel_ = max(pp.v_slow,min(agent_vel_,pp.v_fast))

    ## Calculate deltas ##
    dvs = Array(Float64, nb_cars)
    dys = Array(Float64, nb_cars)

    # agent
    dvs[1] = a.acc*dt
    dys[1] = a.lane_change

    for i in 2:nb_cars
        
    end

    ## Consistency checking ##
    sorted_inds = sort!(collect(1:nb_cars), by=i->s.env_cars[i].pos[1]) # this might be slow because anonymous functions are slow

    for i_unsorted in 1:nb_cars-1
        i = sorted_inds[i_unsorted]
        j = sorted_inds[i_unsorted+1]
        car_i = s.env_cars[i]
        car_j = s.env_cars[j]

        # check if both cars are trying to change into the same lane at the same spot

        # check if they are both starting to change lanes
        if dys[i] != 0.0 && dys[j] != 0.0 && isinteger(car_i.pos[2]) && isinteger(car_j.pos[2])

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

    ## Dynamics


    for (i,car) in enumerate(s.env_cars)
        
        @assert car.behavior.rationality == 1.0 # ignore this for now

        if car.pos[1] >= 0.
            pos = car.pos
            vel = car.vel
            lane_change = car.lane_change
            lane_ = max(1,min(pos[2]+lane_change,nb_col))

            #XXX need to take into account the ego vehicle
            neighborhood = get_adj_cars(pp, s.env_cars, i)

            #call idm model
            dvel_ms = get_idm_dv(car.behavior.p_idm,
                                 dt, vel,
                                 get(neighborhood.ahead_dv, 0, 0.),
                                 get(neighborhood.ahead_dist, 0, 1000.))

            dvel_ms = min(max(dvel_ms/dt, -car.behavior.p_idm.b), car.behavior.p_idm.a)*dt
            vel_ = vel + dvel_ms
            pos_ = pos[1] + dt*(vel-s.agent_vel)


            if (pos_ > pp.lane_length) || (pos_ < 0.)
                push!(car_states, CarState((-1.,1),1.,0,BEHAVIORS[1]))
                continue
            end

            vel_ = max(min(vel_, pp.v_max), pp.v_min)

            #sample lanechange
            #TODO add safety check here
            if mod(lane_,2) == 0 #in between lanes
                @assert lane_change != 0 # shouldn't be in-between lanes if this is zero

            else
                #sample normally
                lanechange_ = get_mobil_lane_change(pp, car, neighborhood)
                #if frnot neighbor is lanechanging, don't lane change
                if neighborhood.ahead_dv != 0 
                    lanechange_ = 0
                end
                lane_change_other = setdiff([-1;0;1],[lanechange_])
                #safety criterion is hard
                if is_lanechange_dangerous(neighborhood,dt,pp.l_car,1)
                    lane_change_other = setdiff(lane_change_other,[1])
                end
                if is_lanechange_dangerous(neighborhood,dt,pp.l_car,-1)
                    lane_change_other = setdiff(lane_change_other,[-1])
                end

                lanechange_other_probs = ((1-car.behavior.rationality)/length(lane_change_other))*ones(length(lane_change_other))
                lanechange_probs = WeightVec([car.behavior.rationality;lanechange_other_probs])
                lanechange = sample(rng,[lanechange_;lane_change_other],lanechange_probs)
                #NO LANECHANGING
                #lanechange = 0
            end

            push!(car_states,CarState((pos_,lane_),vel_,lanechange,car.behavior))
        else
            #TODO: push this to a second loop after this loop
            r = rand(rng)
            if r < 1.-mdp.dmodel.encounter_prob
                push!(encounter_inds,i)
                #continue
            end
            push!(car_states,car)
        end
    end
    =#

end

# """
# Detect if two cars are moving into the same lane.
# 
# Assumes that i is the index of a car that is ahead of or even with j longitudinally
# """
# function lanechange_conflict(state::MLState, i::Int, j::Int)
#     
# end
