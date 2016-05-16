actions(mdp::OriginalMDP) = ActionSpace([MLAction(x[1],x[2]) for x in product(mdp.dmodel.accels,[-1,0,1])])
actions(mdp::OriginalMDP,s::MLState,A::ActionSpace=actions(mdp)) = A #SARSOP does not support partially available actions

function __reward(mdp::OriginalMDP,s::MLState,a::MLAction)

    pos = s.env_cars[1].y#s.agent_pos
    acc = a.acc
    lane_change = a.lane_change
        nb_col = 2*mdp.dmodel.phys_param.nb_lanes-1

    cost = 0.
    if a.acc > 0
        cost += mdp.rmodel.accel_cost*a.acc
    elseif a.acc < 0
        cost -= mdp.rmodel.decel_cost*a.acc
    end
    if abs(a.lane_change) != 0
        cost += mdp.rmodel.lanechange_cost
        if (a.lane_change == -1) && (pos == 1)
            cost += mdp.rmodel.invalid_cost
        elseif (a.lane_change == 1) && (pos == nb_col)
            cost += mdp.rmodel.invalid_cost
        end
    end
    if mod(pos,2) == 0 #on a lane divider--an
        cost += mdp.rmodel.lineride_cost
    end


    return cost #
end

function reward(mdp::OriginalMDP,s::MLState,a::MLAction,sp::MLState)
    if sp.crashed
        return mdp.rmodel.r_crash
    end
    return __reward(mdp,s,a)
end

discount(mdp::Union{MLMDP,MLPOMDP}) = mdp.discount
isterminal(mdp::Union{MLMDP,MLPOMDP},s::MLState) = s.crashed

function GenerativeModels.initial_state(mdp::OriginalMDP, rng::AbstractRNG=MersenneTwister(34985))
   s0 = MLState(false,CarState[CarState(-1., 1, 1., 0, mdp.dmodel.BEHAVIORS[1],0) for i = 1:mdp.dmodel.nb_cars])
   #insert ego car at index 1
   insert!(s0.env_cars,1,CarState(mdp.dmodel.phys_param.lane_length/2., 1., mdp.dmodel.phys_param.v_med, 0, Nullable{BehaviorModel}(), 0))
   return s0
end


#NOTE: from StatsBase
function sample(rng::AbstractRNG,wv::WeightVec)
    t = rand(rng) * sum(wv)
    w = values(wv)
    n = length(w)
    i = 1
    cw = w[1]
    while cw < t && i < n
            i += 1
            @inbounds cw += w[i]
    end
    return i
end
sample(rng::AbstractRNG,a::AbstractArray, wv::WeightVec) = a[sample(rng,wv)]

function generate_s(mdp::OriginalMDP, s::MLState, a::MLAction, rng::AbstractRNG, sp::MLState=create_state(mdp))

    pp = mdp.dmodel.phys_param
    dt = pp.dt
    nb_col = 2*pp.nb_lanes-1
    BEHAVIORS = mdp.dmodel.BEHAVIORS
    ##agent position always updates deterministically
    agent_vel = s.env_cars[1].vel
    agent_lane_ = s.env_cars[1].y + a.lane_change
    agent_lane_ = round(max(1,min(agent_lane_,nb_col)))#can't leave the grid, snap to lane

    agent_vel_ = agent_vel + a.acc*dt
    #underactuating
    agent_vel_ = max(pp.v_slow,min(agent_vel_,pp.v_fast))

    car_states = resize!(sp.env_cars,0)
    push!(car_states,CarState(s.env_cars[1].x, agent_lane_,
                                agent_vel_,s.env_cars[1].lane_change,
                                Nullable{BehaviorModel}()), 0)
    valid_col_top = collect(1:2:nb_col)
    valid_col_bot = collect(1:2:nb_col)

    encounter_inds = Int[] #which cars need to be updated in the second loop

    for (i,car) in enumerate(s.env_cars)
        if i == 1
          continue
        end
        if car.x >= 0.
            x = car.x
            y = car.y
            vel = car.vel
            lane_change = car.lane_change
            behavior = get(car.behavior)

            neighborhood = get_neighborhood(pp,s,i)

            lane_ = round(max(1,min(y+lane_change,nb_col)))
            pos_ = x + dt*(vel-agent_vel)
            if (pos_ > pp.lane_length) || (pos_ < 0.)
                push!(car_states,CarState(-1.,1,1.,0,BEHAVIORS[1],0))
                continue
            end
            #if in between lanes, continue lanechange with prob behavior.rationality, else go other direction

            vel_ = vel + dt*generate_accel(behavior,mdp.dmodel,s,neighborhood,i,rng)

            vel_ = max(min(vel_,pp.v_max),pp.v_min)
            #sample lanechange
            #if in between lanes, continue lanechange with prob behavior.rationality, else go other direction
            lanechange_ = generate_lane_change(behavior,mdp.dmodel,s,neighborhood,i,rng)

            #if near top, remove from valid_col_top
            if pp.lane_length - pos_ <= pp.l_car*1.5
                remove_set = [lane_;lane_+1;lane_-1]
                for idx in remove_set
                    idy = findfirst(valid_col_top,idx)
                    if idy > 0
                        splice!(valid_col_top,idy)
                    end
                end
            #elseif near bot, remove from valid_col_bot
            elseif pos_ < 1.5*pp.l_car
                remove_set = [lane_;lane_+1;lane_-1]
                for idx in remove_set
                    idy = findfirst(valid_col_bot,idx)
                    if idy > 0
                        splice!(valid_col_bot,idy)
                    end
                end
            end

            push!(car_states,CarState(pos_,lane_,vel_,lanechange_,car.behavior,0))
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

    pos_enter = vcat([(0.,y) for y in valid_col_bot],
                                    [(pp.lane_length,y) for y in valid_col_top])

    # this loop is for initialising new cars
    for j in encounter_inds
        if length(pos_enter) <= 0
            #this should work find since that just means they're unencountered by default
            break
        end
        r = rand(rng,1:length(pos_enter))
        pos = splice!(pos_enter,r)
        if pos[1] > 0.
            vels = (pp.v_min,agent_vel_)#collect(1:agent_vel_)
        else
            vels = (agent_vel_,pp.v_max)#collect(agent_vel_:pp.nb_vel_bins)
        end
        vel = (vels[2]-vels[1])*rand(rng)+vels[1]#vels[rand(rng,1:length(vels))]
        if mod(pos[2],2) == 0
            lanechanges = [-1.;1.]
        else
            lanechanges = [-1.;0.;1.]
        end
        lanechange = lanechanges[rand(rng,1:length(lanechanges))]
        behavior = deepcopy(BEHAVIORS[rand(rng,1:length(BEHAVIORS))])
        behavior.p_idm.v0 = (min(behavior.p_idm.v0+1,pp.v_max)-
                                                    max(behavior.p_idm.v0-1,pp.v_min))*
                                                    rand(rng) + behavior.p_idm.v0

        car_states[j] = CarState(pos[1], pos[2], vel, lanechange, behavior, 0)
    end

    sp.crashed = is_crash(mdp, s, sp)
    @assert sp.env_cars === car_states

    return sp
end

iterator(as::ActionSpace) = as.actions
length(space::ActionSpace) = length(space.actions)
function rand(rng::AbstractRNG, action_space::ActionSpace, a=nothing)
    r = rand(rng, 1:length(action_space))
    return action_space.actions[r]
end

create_state(p::OriginalMDP) = MLState(false, 1, p.dmodel.phys_param.v_med, CarState[CarState(-1.,1,1.,0,p.dmodel.BEHAVIORS[1],0) for _ = 1:p.dmodel.nb_cars])
create_action(p::OriginalMDP) = MLAction()
create_observation(mdp::MLMDP) = MLObs(create_state(mdp))
