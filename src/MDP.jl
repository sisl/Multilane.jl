actions(mdp::OriginalMDP) = ActionSpace([MLAction(x[1],x[2]) for x in product(mdp.dmodel.accels,[-1,0,1])])
actions(mdp::OriginalMDP,s::MLState,A::ActionSpace=actions(mdp)) = A #SARSOP does not support partially available actions

function __reward(mdp::OriginalMDP,s::MLState,a::MLAction)

    pos = s.agent_pos
    vel = a.vel
    lane_change = a.lane_change
        nb_col = 2*mdp.dmodel.phys_param.nb_lanes-1

    cost = 0.
    if a.vel > 0
        cost += mdp.rmodel.accel_cost*a.vel
    elseif a.vel < 0
        cost -= mdp.rmodel.decel_cost*a.vel
    end
    if abs(a.lane_change) != 0
        cost += mdp.rmodel.lanechange_cost
        if (a.lane_change == -1) && (s.agent_pos == 1)
            cost += mdp.rmodel.invalid_cost
        elseif (a.lane_change == 1) && (s.agent_pos == nb_col)
            cost += mdp.rmodel.invalid_cost
        end
    end
    if mod(s.agent_pos,2) == 0 #on a lane divider--an
        cost += mdp.rmodel.lineride_cost
    end


    return cost #
end

# function reward(mdp::OriginalMDP,s::MLState,a::MLAction)
#     #assume environment cars don't crash with one another?
#     if is_crash(mdp,s,a)
#         return mdp.rmodel.r_crash
#     end
#     #penalty for accelerating/decelerating/lane change
# 
#     return __reward(mdp,s,a) #
# end

function reward(mdp::OriginalMDP,s::MLState,a::MLAction,sp::MLState)
    if sp.crashed
        return mdp.rmodel.r_crash
    end
    return __reward(mdp,s,a)
end

discount(mdp::MLMDP) = mdp.discount
isterminal(mdp::MLMDP,s::MLState) = s.crashed

function GenerativeModels.initial_state(mdp::OriginalMDP, rng::AbstractRNG=MersenneTwister(34985))
    return MLState(false, 1, mdp.dmodel.phys_param.v_med,
                                     CarState[CarState((-1.,1),1.,0,mdp.dmodel.BEHAVIORS[1]) for i = 1:mdp.dmodel.nb_cars])
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

function GenerativeModels.generate_s(mdp::OriginalMDP, s::MLState, a::MLAction, rng::AbstractRNG, sp::MLState=create_state(mdp))

    pp = mdp.dmodel.phys_param
    dt = pp.dt
    nb_col = 2*pp.nb_lanes-1
    BEHAVIORS = mdp.dmodel.BEHAVIORS
    ##agent position always updates deterministically
    agent_lane_ = s.agent_pos + a.lane_change
    agent_lane_ = max(1,min(agent_lane_,nb_col)) #can't leave the grid

    agent_vel_ = s.agent_vel + a.vel*dt#convert(Int,ceil(a.vel*dt/(v_interval)))
    #underactuating
    agent_vel_ = max(pp.v_slow,min(agent_vel_,pp.v_fast))

    car_states = CarState[]
    valid_col_top = collect(1:2:nb_col)
    valid_col_bot = collect(1:2:nb_col)

    encounter_inds = Int[] #which cars need to be updated in the second loop

    for (i,car) in enumerate(s.env_cars)
        if car.pos[1] >= 0.
            pos = car.pos
            vel = car.vel
            lane_change = car.lane_change
            lane_ = max(1,min(pos[2]+lane_change,nb_col))

            neighborhood = get_adj_cars(pp,s.env_cars,i)

            dvel_ms = get_idm_dv(car.behavior.p_idm,dt,vel,get(neighborhood.ahead_dv,0,0.),get(neighborhood.ahead_dist,0,1000.)) #call idm model
            #bound by the acceleration/braking terms in idm models
            #NOTE restricting available velocities
            dvel_ms = min(max(dvel_ms/dt,-car.behavior.p_idm.b),car.behavior.p_idm.a)*dt
            vel_ = vel + dvel_ms
            pos_ = pos[1] + dt*(vel-s.agent_vel)
            if (pos_ > pp.lane_length) || (pos_ < 0.)
                push!(car_states,CarState((-1.,1),1.,0,BEHAVIORS[1]))
                continue
            end
            #TODO add noise to position transition

            #sample velocity
            #accelerate normally or dont accelerate
            if rand(rng) < 1- car.behavior.rationality
                vel_ = vel
            end
            vel_ = max(min(vel_,pp.v_max),pp.v_min)
            #sample lanechange
            #if in between lanes, continue lanechange with prob behavior.rationality, else go other direction
            #TODO add safety check here
            if mod(lane_,2) == 0 #in between lanes
                r = rand(rng)
                #if on the off chance its not changing lanes, make it, the jerk
                if lane_change == 0
                    lane_change = rand(rng,-1:2:1)
                end
                lanechange = r < car.behavior.rationality ? lane_change : -1*lane_change

                if is_lanechange_dangerous(neighborhood,dt,pp.l_car,lanechange)
                    lanechange *= -1
                end

            else
                #sample normally
                lanechange_ = get_mobil_lane_change(pp,car,neighborhood)
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

    pos_enter = vcat([(0.,y) for y in valid_col_bot],
                                    [(pp.lane_length,y) for y in valid_col_top])
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
            #in between lanes, so he must be lane changing
            lanechanges = [-1;1]
        else
            lanechanges = [-1;0;1]
        end
        lanechange = lanechanges[rand(rng,1:length(lanechanges))]
        behavior = deepcopy(BEHAVIORS[rand(rng,1:length(BEHAVIORS))])
        behavior.p_idm.v0 = (min(behavior.p_idm.v0+1,pp.v_max)-
                                                    max(behavior.p_idm.v0-1,pp.v_min))*
                                                    rand(rng) + behavior.p_idm.v0

        car_states[j] = CarState(pos,vel,lanechange,behavior)
        #push!(car_states,CarState(pos,vel,lanechange,behavior))
    end

    return MLState(is_crash(mdp, s, a), agent_lane_, agent_vel_, car_states)

end

iterator(as::ActionSpace) = as.actions
length(space::ActionSpace) = length(space.actions)
function rand(rng::AbstractRNG, action_space::ActionSpace, a=nothing)
    r = rand(rng, 1:length(action_space))
    return action_space.actions[r]
end

create_state(p::OriginalMDP) = MLState(false, 1, p.dmodel.phys_param.v_med, CarState[CarState((-1.,1,),1.,0,p.dmodel.BEHAVIORS[1]) for _ = 1:p.dmodel.nb_cars])
create_action(p::OriginalMDP) = MLAction()
