mutable struct NoCrashRewardModel <: AbstractMLRewardModel
    cost_dangerous_brake::Float64 # POSITIVE NUMBER
    reward_in_target_lane::Float64 # POSITIVE NUMBER

    brake_penalty_thresh::Float64 # (POSITIVE NUMBER) if the deceleration is greater than this cost_dangerous_brake will be accured
    target_lane::Int
end

#XXX temporary
NoCrashRewardModel() = NoCrashRewardModel(100.,10.,2.5,4)
lambda(rm::NoCrashRewardModel) = rm.cost_dangerous_brake/rm.reward_in_target_lane

mutable struct NoCrashIDMMOBILModel{G<:BehaviorGenerator} <: AbstractMLDynamicsModel
    nb_cars::Int
    phys_param::PhysicalParam

    # behaviors::Vector{BehaviorModel}
    # behavior_probabilities::Weights
    behaviors::G

    adjustment_acceleration::Float64
    lane_change_rate::Float64 # in LANES PER SECOND

    p_appear::Float64 # probability of a new car appearing if the maximum number are not on the road
    appear_clearance::Float64 # minimum clearance for a car to appear

    vel_sigma::Float64 # std of new car speed about v0

    lane_terminate::Bool # if true, terminate the simulation when the car has reached the desired lane
    brake_terminate_thresh::Float64 # terminate simulation if braking is above this (always positive)
    max_dist::Float64 # terminate simulation if distance becomes greater than this
    speed_terminate_thresh::Float64 # terminate simulation if any speed is below this
end

function NoCrashIDMMOBILModel(nb_cars::Int,
                              pp::PhysicalParam;
                              vel_sigma = 0.5,
                              lane_terminate=false,
                              p_appear=0.5,
                              brake_terminate_thresh=Inf,
                              max_dist=Inf,
                              speed_terminate_thresh=-Inf,
                              behaviors=DiscreteBehaviorSet(IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in
                                                 enumerate(Iterators.product(["cautious","normal","aggressive"],
                                                        [pp.v_slow+0.5;pp.v_med;pp.v_fast],
                                                        [pp.l_car]))], Weights(ones(9)))
                              )

    return NoCrashIDMMOBILModel(
        nb_cars,
        pp,
        behaviors,
        1., # adjustment accel
        1.0/(2.0*pp.dt), # lane change rate
        p_appear, # p_appear
        35.0, # appear_clearance
        vel_sigma, # vel_sigma
        lane_terminate,
        brake_terminate_thresh,
        max_dist,
        speed_terminate_thresh
    )
end

function NoCrashIDMMOBILModel(old::NoCrashIDMMOBILModel, behaviors::BehaviorGenerator)
    return NoCrashIDMMOBILModel(old.nb_cars, old.phys_param, behaviors, old.adjustment_acceleration, old.lane_change_rate, old.p_appear, old.appear_clearance, old.vel_sigma, old.lane_terminate, old.brake_terminate_thresh, old.max_dist, old.speed_terminate_thresh)
end

const NoCrashMDP{R<:AbstractMLRewardModel, G} =  MLMDP{MLState, MLAction, NoCrashIDMMOBILModel{G}, R}
const NoCrashPOMDP{R<:AbstractMLRewardModel, G} =  MLPOMDP{MLState, MLAction, MLObs, NoCrashIDMMOBILModel{G}, R}

const NoCrashProblem{R<:AbstractMLRewardModel,G} =  Union{NoCrashMDP{R,G}, NoCrashPOMDP{R,G}}

gen_type(::Type{MLPOMDP{S,A,O,NoCrashIDMMOBILModel{G},R}}) where {S,A,O,G,R} = G

# create_action(::NoCrashProblem) = MLAction()

# action space = {a in {accelerate,maintain,decelerate}x{left_lane_change,maintain,right_lane_change} | a is safe} U {brake}
struct NoCrashActionSpace
    NORMAL_ACTIONS::Vector{MLAction} # all the actions except brake
    acceptable::IntSet
    brake::MLAction # this action will be EITHER braking at half the dangerous brake threshold OR the braking necessary to prevent a collision at all time in the future
end

const NB_NORMAL_ACTIONS = 9

function NoCrashActionSpace(mdp::NoCrashProblem)
    accels = (-mdp.dmodel.adjustment_acceleration, 0.0, mdp.dmodel.adjustment_acceleration)
    lane_changes = (-mdp.dmodel.lane_change_rate, 0.0, mdp.dmodel.lane_change_rate)
    NORMAL_ACTIONS = MLAction[MLAction(a,l) for (a,l) in Iterators.product(accels, lane_changes)] # this should be in the problem
    return NoCrashActionSpace(NORMAL_ACTIONS, IntSet(), MLAction()) # note: brake will be calculated later based on the state
end

function actions(mdp::NoCrashProblem, s::Union{MLState, MLPhysicalState})
    as = NoCrashActionSpace(mdp)
    acceptable = IntSet()
    for i in 1:NB_NORMAL_ACTIONS
        a = as.NORMAL_ACTIONS[i]
        ego_y = s.cars[1].y
        # prevent going off the road
        if ego_y == 1. && a.lane_change < 0. || ego_y == mdp.dmodel.phys_param.nb_lanes && a.lane_change > 0.0
            continue
        end
        # prevent running into the person in front or to the side
        if is_safe(mdp, s, as.NORMAL_ACTIONS[i])
            push!(acceptable, i)
        end
    end
    brake_acc = calc_brake_acc(mdp, s)
    brake = MLAction(brake_acc, 0)
    return NoCrashActionSpace(as.NORMAL_ACTIONS, acceptable, brake)
end

actions(mdp::NoCrashProblem, b::AbstractParticleBelief) = actions(mdp, MLPhysicalState(first(particles(b))))

calc_brake_acc(mdp::NoCrashProblem, s::Union{MLState, MLPhysicalState}) = min(max_safe_acc(mdp,s), -mdp.rmodel.brake_penalty_thresh/2.0)
# calc_brake_acc(mdp::NoCrashProblem{TargetLaneReward}, s::Union{MLState, MLPhysicalState}) = min(max_safe_acc(mdp, s), min(-mdp.dmodel.phys_param.brake_limit/2, -mdp.dmodel.brake_terminate_thresh/2))

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
function max_safe_acc(mdp::NoCrashProblem, s::Union{MLState,MLObs}, lane_change::Float64=0.0)
    dt = mdp.dmodel.phys_param.dt
    v_min = mdp.dmodel.phys_param.v_min
    l_car = mdp.dmodel.phys_param.l_car
    bp = mdp.dmodel.phys_param.brake_limit
    ego = s.cars[1]

    car_in_front = 0
    smallest_gap = Inf
    # find car immediately in front
    if length(s.cars) > 1
        for i in 2:length(s.cars)#nb_cars
            if occupation_overlap(s.cars[i].y, 0.0, ego.y, lane_change) # occupying same lane
                gap = s.cars[i].x - ego.x - l_car
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
            n_brake_acc = nullable_max_safe_acc(smallest_gap, ego.vel, s.cars[car_in_front].vel, bp, dt)
            return get(n_brake_acc, -mdp.dmodel.phys_param.brake_limit)
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
    # VVV see mathematica notebook
    return - (bp*dt + 2.*v - sqrt(8.*g*bp + bp^2*dt^2 - 4.*bp*dt*v + 4.*vo^2)) / (2.*dt)
end

"""
Return max_safe_acc or an empty Nullable if the discriminant is negative.
"""
function nullable_max_safe_acc(gap, v_behind, v_ahead, braking_limit, dt)
    bp = braking_limit
    v = v_behind
    vo = v_ahead
    g = gap
    # VVV see mathematica notebook
    discriminant = 8.*g*bp + bp^2*dt^2 - 4.*bp*dt*v + 4.*vo^2
    if discriminant >= 0.0
        return Nullable{Float64}(- (bp*dt + 2.*v - sqrt(discriminant)) / (2.*dt))
    else
        return Nullable{Float64}()
    end
end



"""
Test whether, if the ego vehicle takes action a, it will always be able to slow down fast enough if the car in front slams on his brakes and won't pull in front of another car so close they can't stop
"""
function is_safe(mdp::NoCrashProblem, s::Union{MLState,MLObs}, a::MLAction)
    dt = mdp.dmodel.phys_param.dt
    if a.acc >= max_safe_acc(mdp, s, a.lane_change)
        return false
    end
    # check whether we will go into anyone else's lane so close that they might hit us or we might run into them
    if isinteger(s.cars[1].y) && a.lane_change != 0.0
        l_car = mdp.dmodel.phys_param.l_car
        for i in 2:length(s.cars)
            car = s.cars[i]
            ego = s.cars[1]
            if car.x < ego.x + l_car && occupation_overlap(ego.y, a.lane_change, car.y, 0.0)  # ego is in front of car
                # New definition of safe - the car behind can brake at max braking to avoid the ego if the ego
                # slams on his brakes
                # XXX IS THIS RIGHT??
                # I think I need a better definition of "safe" here
                gap = ego.x - car.x - l_car
                if gap <= 0.0
                    return false
                end
                n_braking_acc = nullable_max_safe_acc(gap, car.vel, ego.vel, mdp.dmodel.phys_param.brake_limit, dt)

                # if isnull(n_braking_acc) || get(n_braking_acc) < -mdp.dmodel.phys_param.brake_limit
                if isnull(n_braking_acc) || get(n_braking_acc) < max_accel(mdp.dmodel.behaviors)
                    return false
                end
            end
        end
    end
    return true
end

#XXX temp
create_state(p::NoCrashProblem) = MLState(0.0, 0.0, Vector{CarState}(p.dmodel.nb_cars), nothing)
create_observation(pomdp::NoCrashPOMDP) = MLObs(0.0, 0.0, Vector{CarStateObs}(pomdp.dmodel.nb_cars), nothing)

function generate_s(mdp::NoCrashProblem, s::MLState, a::MLAction, rng::AbstractRNG)

    @if_debug dbg_rng = copy(rng)

    sp::MLState=create_state(mdp)

    try
        pp = mdp.dmodel.phys_param
        dt = pp.dt
        nb_cars = length(s.cars)
        resize!(sp.cars, nb_cars)
        sp.terminal = s.terminal
        sp.t = s.t + dt

        ## Calculate deltas ##
        #====================#

        dxs = Vector{Float64}(nb_cars)
        dvs = Vector{Float64}(nb_cars)
        dys = Vector{Float64}(nb_cars)
        lcs = Vector{Float64}(nb_cars)

        # agent
        dvs[1] = a.acc*dt
        dxs[1] = s.cars[1].vel*dt + a.acc*dt^2/2.
        lcs[1] = a.lane_change
        dys[1] = a.lane_change*dt

        for i in 2:nb_cars
            neighborhood = get_neighborhood(pp, s, i)

            behavior = s.cars[i].behavior

            acc = gen_accel(behavior, mdp.dmodel, s, neighborhood, i, rng)
            dvs[i] = dt*acc
            dxs[i] = (s.cars[i].vel + dvs[i]/2.)*dt

            lcs[i] = gen_lane_change(behavior, mdp.dmodel, s, neighborhood, i, rng)
            dys[i] = lcs[i] * dt
        end

        ## Consistency checking ##
        #========================#

        # first prevent lane changes into each other
        changers = IntSet()
        for i in 1:nb_cars
            if lcs[i] != 0
                push!(changers, i)
            end
        end
        sorted_changers = sort!(collect(changers), by=i->s.cars[i].x, rev=true) # this might be slow because anonymous functions are slow
        # from front to back

        if length(sorted_changers) >= 2 #something to compare
            # iterate through pairs
            iter_state = start(sorted_changers)
            j, iter_state = next(sorted_changers, iter_state)
            while !done(sorted_changers, iter_state)
                i = j
                j, iter_state = next(sorted_changers, iter_state)
                car_i = s.cars[i] # front
                car_j = s.cars[j] # back

                # check if they are both starting to change lanes on this step
                if isinteger(car_i.y) && isinteger(car_j.y)

                    # check if they are near each other lanewise
                    if abs(car_i.y - car_j.y) <= 2.0

                        # make sure there is a conflict longitudinally
                        # if car_i.x - car_j.x <= pp.l_car || car_i.x + dxs[i] - car_j.x + dxs[j] <= pp.l_car
                        # if car_i.x - car_j.x <= mdp.dmodel.appear_clearance # made more conservative on 8/19
                        # if car_i.x - car_j.x <= get_idm_s_star(car_j.behavior.p_idm, car_j.vel, car_j.vel-car_i.vel) # upgraded to sstar on 8/19
                        ivp = car_i.vel + dt*dvs[i]
                        jvp = car_j.vel + dt*dvs[j]
                        ixp = car_i.x + dt*(car_i.vel + ivp)/2.0
                        jxp = car_j.x + dt*(car_j.vel + jvp)/2.0
                        n_max_acc_p = nullable_max_safe_acc(ixp-jxp-pp.l_car, jvp, ivp, pp.brake_limit,dt)
                        if ixp - jxp <= pp.l_car || car_i.x - car_j.x <= pp.l_car || isnull(n_max_acc_p) || get(n_max_acc_p) < -pp.brake_limit

                            # check if they are moving towards each other
                            # if dys[i]*dys[j] < 0.0 && abs(car_i.y+dys[i] - car_j.y+dys[j]) < 2.0
                            if true # prevent lockstepping 8/19 (doesn't prevent it that well)

                                # make j stay in his lane
                                dys[j] = 0.0
                                lcs[j] = 0.0
                            end
                        end
                    end
                end
            end
        end

        # second, prevent cars hitting each other due to noise
        if mdp.throw
            # sorted = sort!(collect(1:length(s.cars)), by=i->s.cars[i].x, rev=true)
            sorted = sortperm(collect(c.x for c in s.cars), rev=true)

            if length(sorted) >= 2 #something to compare
                # iterate through pairs
                iter_state = start(sorted)
                j, iter_state = next(sorted, iter_state)
                while !done(sorted, iter_state)
                    i = j
                    j, iter_state = next(sorted, iter_state)
                    if j == 1
                        continue # don't check for the ego since the ego does not have noise
                    end
                    car_i = s.cars[i]
                    car_j = s.cars[j]

                    # check if they overlap longitudinally
                    if car_j.x + dxs[j] > car_i.x + dxs[i] - pp.l_car
                    
                        # check if they will be in the same lane
                        if occupation_overlap(car_i.y + dys[i], car_j.y + dys[j])
                            # warn and nudge behind
                            if mdp.throw
                                @show car_j.x
                                @show car_i.x
                                @show car_j.x + dxs[j]
                                @show car_i.x + dxs[i]
                                @show pp.l_car
                            end
                            @if_debug begin
                                println("Conflict because of noise: front:$i, back:$j")
                                # Gallium.@enter generate_s(mdp, s, a, dbg_rng)
                                fname = tempname()*".jld"
                                println("saving debug args to $fname")
                                JLD.@save(fname, mdp, s, a, dbg_rng)
                            end
                            if i == 1
                                if mdp.throw
                                    error("Car nudged because of crash (ego in front).")
                                end
                            else
                                # warn("Car nudged because noise would cause a crash.")
                                if mdp.throw
                                    error("Car nudged because of crash.")
                                end
                            end
                            dxs[j] = car_i.x + dxs[i] - car_j.x - 1.01*pp.l_car
                            dvs[j] = 2.0*(dxs[j]/dt - car_j.vel)
                        end
                    end
                end
            end
        end

        ## Dynamics and Exits ##
        #======================#

        exits = IntSet()
        for i in 1:nb_cars
            car = s.cars[i]
            xp = car.x + (dxs[i] - dxs[1])
            yp = car.y + dys[i]
            # velp = max(min(car.vel + dvs[i],pp.v_max), pp.v_min)
            velp = max(car.vel + dvs[i], 0.0) # removed speed limits on 8/13
            # note lane change is updated above

            if velp < mdp.dmodel.speed_terminate_thresh
                sp.terminal = Nullable{Symbol}(:speed)
            end
            if dvs[i]/dt < -mdp.dmodel.brake_terminate_thresh
                sp.terminal = Nullable{Symbol}(:brake)
            end

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

            # if yp < 1.0 || yp > pp.nb_lanes
            #     @show i
            #     @show yp
            #     println("mdp = $mdp")
            #     println("s = $s")
            #     println("a = $a")
            # end
            @assert yp >= 1.0 && yp <= pp.nb_lanes

            if xp < 0.0 || xp >= pp.lane_length
                push!(exits, i)
            else
                sp.cars[i] = CarState(xp, yp, velp, lcs[i], car.behavior, s.cars[i].id)
            end
        end

        sp.x = s.x + dxs[1]

        next_id = maximum([c.id for c in s.cars]) + 1

        deleteat!(sp.cars, exits)
        nb_cars -= length(exits)

        ## Generate new cars ##
        #=====================#

        if nb_cars < mdp.dmodel.nb_cars && rand(rng) <= mdp.dmodel.p_appear

            behavior = rand(rng, mdp.dmodel.behaviors)
            vel = typical_velocity(behavior) + randn(rng)*mdp.dmodel.vel_sigma

            clearances = Vector{Float64}(pp.nb_lanes)
            fill!(clearances, Inf)
            closest_cars = Vector{Int}(pp.nb_lanes)
            fill!(closest_cars, 0)
            sstar_margins = Vector{Float64}(pp.nb_lanes)
            if vel > sp.cars[1].vel
                # put at back
                # sstar is the sstar of the new guy
                for i in 1:length(sp.cars)
                    lowlane, highlane = occupation_lanes(sp.cars[i].y, 0.0)
                    back = sp.cars[i].x - pp.l_car
                    if back < clearances[lowlane]
                        clearances[lowlane] = back
                        closest_cars[lowlane] = i
                    end
                    if back < clearances[highlane]
                        clearances[highlane] = back
                        closest_cars[highlane] = i
                    end
                end
                for j in 1:pp.nb_lanes
                    other = closest_cars[j]
                    if other == 0
                        sstar = 0.0
                    else
                        sstar = get_idm_s_star(behavior.p_idm::IDMParam, vel, vel-sp.cars[other].vel::Float64)
                    end
                    sstar_margins[j] = clearances[j] - sstar
                end
            else
                for i in 1:length(sp.cars)
                    lowlane, highlane = occupation_lanes(sp.cars[i].y, 0.0)
                    front = pp.lane_length - (sp.cars[i].x + pp.l_car) # l_car is half the length of the old car plus half the length of the new one
                    if front < clearances[lowlane]
                        clearances[lowlane] = front
                        closest_cars[lowlane] = i
                    end
                    if front < clearances[highlane]
                        clearances[highlane] = front
                        closest_cars[highlane] = i
                    end
                end
                for j in 1:pp.nb_lanes
                    other = closest_cars[j]
                    if other == 0
                        sstar = 0
                    else
                        sstar = get_idm_s_star(sp.cars[other].behavior.p_idm,
                                               sp.cars[other].vel,
                                               sp.cars[other].vel-vel)
                    end
                    sstar_margins[j] = clearances[j] - sstar
                end
            end

            margin, lane = findmax(sstar_margins)
            
            if margin > 0.0
                if vel > sp.cars[1].vel
                    # at back
                    push!(sp.cars, CarState(0.0, lane, vel, 0.0, behavior, next_id))
                else
                    push!(sp.cars, CarState(pp.lane_length, lane, vel, 0.0, behavior, next_id))
                end
            end
        end

        if sp.x > mdp.dmodel.max_dist
            sp.terminal = Nullable{Any}(:distance)
        elseif mdp.dmodel.lane_terminate && sp.cars[1].y == mdp.rmodel.target_lane
            sp.terminal = Nullable{Any}(:lane)
        end

        @assert sp.cars[1].x == s.cars[1].x # ego should not move
        @if_debug begin
            if any(any(isnan.(c.behavior.p_idm)) for c in sp.cars)
                warn("NaN in idm.")
            end
        end
    catch ex
        if mdp.throw
            rethrow(ex)
        else
            sp = deepcopy(s)
            sp.terminal = Nullable{Any}(ex)
            return sp
        end
    end

    return sp
end

function reward(mdp::NoCrashProblem{NoCrashRewardModel}, s::MLState, ::MLAction, sp::MLState)
    r = 0.0
    if sp.cars[1].y == mdp.rmodel.target_lane
        r += mdp.rmodel.reward_in_target_lane
    end
    nb_brakes = detect_braking(mdp, s, sp)
    r -= mdp.rmodel.cost_dangerous_brake*nb_brakes
    return r
end

"""
Return the number of braking actions that occured during this state transition
"""
function detect_braking(mdp::NoCrashProblem, s::MLState, sp::MLState, threshold::Float64)
    nb_brakes = 0
    nb_leaving = 0 
    dt = mdp.dmodel.phys_param.dt
    for (i,c) in enumerate(s.cars)
        if length(sp.cars) >= i-nb_leaving
            cp = sp.cars[i-nb_leaving]
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

detect_braking(mdp::NoCrashProblem, s::MLState, sp::MLState) = detect_braking(mdp, s, sp, mdp.rmodel.brake_penalty_thresh)

"""
Return the ids of cars that brake during this state transition
"""
function braking_ids(mdp::NoCrashProblem, s::MLState, sp::MLState, threshold=mdp.rmodel.brake_penalty_thresh)
    braking = Int[]
    nb_leaving = 0 
    dt = mdp.dmodel.phys_param.dt
    for (i,c) in enumerate(s.cars)
        if length(sp.cars) >= i-nb_leaving
            cp = sp.cars[i-nb_leaving]
        else
            break
        end
        if cp.id != c.id
            nb_leaving += 1
            continue
        else
            acc = (cp.vel-c.vel)/dt
            if acc < -threshold
                push!(braking, c.id)
            end
        end
    end
    # @assert nb_leaving <= 5 # sanity check - can remove this if it is violated as long as it doesn't happen all the time
    return braking
end

"""
Return the maximum braking (a positive number) by any car during the transition between s an sp
"""
function max_braking(mdp::NoCrashProblem, s::MLState, sp::MLState)
    nb_leaving = 0 
    dt = mdp.dmodel.phys_param.dt
    min_acc = 0.0
    for (i,c) in enumerate(s.cars)
        if length(sp.cars) >= i-nb_leaving
            cp = sp.cars[i-nb_leaving]
        else
            break
        end
        if cp.id != c.id
            nb_leaving += 1
            continue
        else
            acc = (cp.vel-c.vel)/dt
            if acc < min_acc
                min_acc = acc
            end
        end
    end
    # @assert nb_leaving <= 5 # sanity check - can remove this if it is violated as long as it doesn't happen all the time
    return -min_acc
end

"""
Assign behaviors to a given physical state.
"""
function initial_state(mdp::NoCrashProblem, ps::MLPhysicalState, rng::AbstractRNG)
    s = MLState(ps, Vector{CarState}(length(ps.cars)))
    s.cars[1] = CarState(ps.cars[1], NORMAL)
    for i in 2:length(s.cars)
        behavior = rand(rng, mdp.dmodel.behaviors)
        s.cars[i] = CarState(ps.cars[i], behavior)
    end
    s.x = 0.0
    s.t = 0.0
    return s
end

function initial_state(p::NoCrashProblem, rng::AbstractRNG=Base.GLOBAL_RNG)
    @if_debug println("debugging")
    mdp = NoCrashMDP{typeof(p.rmodel), typeof(p.dmodel.behaviors)}(p.dmodel, p.rmodel, p.discount, p.throw) # make sure an MDP
    return relaxed_initial_state(mdp, 200, rng)
end

function generate_o(mdp::NoCrashProblem, s::MLState, a::MLAction, sp::MLState)
    return MLObs(sp)
end

function generate_sor(pomdp::NoCrashPOMDP, s::MLState, a::MLAction, rng::AbstractRNG)
    sp, r = generate_sr(pomdp, s, a, rng)
    o = generate_o(pomdp, s, a, sp)
    return sp, o, r
end

@if_debug function generate_sori(pomdp::NoCrashPOMDP, s::MLState, a::MLAction, rng::AbstractRNG)
    rngcp = copy(rng)
    sp, o, r = generate_sor(pomdp, s, a, rng)
    return sp, o, r, Dict(:rng=>rngcp)
end

@if_debug function generate_sri(mdp::NoCrashProblem, s, a, rng::AbstractRNG)
    rngcp = copy(rng)
    sp, r = generate_sr(mdp, s, a, rng)
    return sp, r, Dict(:rng=>rngcp)
end

discount(mdp::Union{MLMDP,MLPOMDP}) = mdp.discount
isterminal(mdp::Union{MLMDP,MLPOMDP},s::MLState) = !isnull(s.terminal)
