type IDMMOBILBehavior <: BehaviorModel
	p_idm::IDMParam
	p_mobil::MOBILParam
	rationality::Float64
	idx::Int
end
==(a::IDMMOBILBehavior,b::IDMMOBILBehavior) = (a.p_idm==b.p_idm) && (a.p_mobil==b.p_mobil) &&(a.rationality == b.rationality)
Base.hash(a::IDMMOBILBehavior,h::UInt64=zero(UInt64)) = hash(a.p_idm,hash(a.p_mobil,hash(a.rationality,h)))

function IDMMOBILBehavior(s::AbstractString,v0::Float64,s0::Float64,idx::Int)
	typedict = Dict{AbstractString,Float64}("cautious"=>1.,"normal"=>1.,"aggressive"=>1.) #rationality
	return IDMMOBILBehavior(IDMParam(s,v0,s0),MOBILParam(s),typedict[s],idx)
end

get_dv(bmodel::BehaviorModel, dmodel::AbstractMLDynamicsModel, s::MLState, neighborhood::Array{Int,1}, idx::Int, rng::AbstractRNG) = error("Uninstantiated Behavior Model")

get_dy(bmodel::BehaviorModel, dmodel::AbstractMLDynamicsModel, s::MLState,neighborhood::Array{Int,1}, idx::Int, rng::AbstractRNG) = error("Uninstantiated Behavior Model")


function generate_accel(bmodel::IDMMOBILBehavior, dmodel::IDMMOBILModel, s::MLState, neighborhood::Array{Int,1}, idx::Int, rng::AbstractRNG)
	pp = dmodel.phys_param
	dt = pp.dt
	car = s.env_cars[idx]
	vel = car.vel

	nbr = neighborhood[2]

	dv = nbr != 0 ? vel - s.env_cars[nbr].vel : 0.
	ds = nbr != 0 ? s.env_cars[nbr].pos[1] - car.pos[1] + pp.l_car : 1000.

	dvel = get_idm_dv(get(car.behavior).p_idm,dt,vel,dv,ds) #call idm model
	dvel = min(max(dvel/dt,-get(car.behavior).p_idm.b),get(car.behavior).p_idm.a)
	#accelerate normally or dont accelerate
	if rand(rng) < 1- get(car.behavior).rationality
			dvel = 0
	end
	#make sure it wont result in an inconsistent thing?
	return dvel
end

function generate_lane_change(bmodel::IDMMOBILBehavior, dmodel::IDMMOBILModel, s::MLState,neighborhood::Array{Int,1}, idx::Int, rng::AbstractRNG)

	pp = dmodel.phys_param
	dt = pp.dt
	car = s.env_cars[idx]
	lane_change = car.lane_change
	lane_ = round(max(1,min(car.pos[2]+lane_change,nb_col)))

	if mod(lane_,2) == 0. #in between lanes
		r = rand(rng)
		#if on the off chance its not changing lanes, make it, the jerk
		if lane_change == 0.
				lane_change = rand(rng,-1:2:1)
		end
		lanechange = r < get(car.behavior).rationality ? lane_change : -1*lane_change

		if is_lanechange_dangerous(neighborhood,dt,pp.l_car,lanechange)
				lanechange *= -1.
		end

		return lanechange
	end
	#sample normally
	lanechange_ = get_mobil_lane_change(pp,car,neighborhood)
	#if frnot neighbor is lanechanging, don't lane change
	nbr = neighborhood[2]
	ahead_dy = nbr != 0 ? s.env_cars[nbr].lane_change : 0
	if ahead_dy != 0
			lanechange_ = 0.
	end
	lane_change_other = setdiff([-1;0;1],[lanechange_])
	#safety criterion is hard
	if is_lanechange_dangerous(neighborhood,dt,pp.l_car,1)
			lane_change_other = setdiff(lane_change_other,[1])
	end
	if is_lanechange_dangerous(neighborhood,dt,pp.l_car,-1)
			lane_change_other = setdiff(lane_change_other,[-1])
	end

	lanechange_other_probs = ((1-get(car.behavior).rationality)/length(lane_change_other))*ones(length(lane_change_other))
	lanechange_probs = WeightVec([get(car.behavior).rationality;lanechange_other_probs])
	lanechange = sample(rng,[lanechange_;lane_change_other],lanechange_probs)
	#NO LANECHANGING
	#lanechange = 0

	return lanechange
end

#############################################################################
#TODO How to make compatible with MOBIL?

type AvoidModel <: BehaviorModel
	jerk::Bool
end
AvoidModel() = AvoidModel(false)
# NOTE: JerkModel might be better modeled by a /very/ aggressive IDM car
JerkModel() = AvoidModel(true)
==(a::AvoidModel, b::AvoidModel) = (a.jerk == b.jerk)
Base.hash(a::AvoidModel, h::UInt64=zero(UInt64)) = hash(a.jerk,h)

function closest_car(dmodel::IDMMOBILModel, s::MLState, nbhd::Array{Int,1}, idx::Int, lookahead_only::Bool)
	x, y = s.env_cars[idx].pos
	dy = dmodel.phys_param.y_interval

	closest = 0
	min_dist = Inf

	for i = 1:3 #front cars
		if nbhd[i] == 0
			continue
		end
		dist = norm([x - car.pos[1]; (y - car.pos[2])*dy])
		if dist < min_dist
			min_dist = dist
			closest = i
		end
	end

	if lookahead_only
		return closest
	end

	for i = 4:6 #rear cars
		if nbhd[i] == 0
			continue
		end
		car = s.env_cars[nbhd[i]]
		dists = norm([x - car.pos[1]; (y - car.pos[2])*dy])
		if dist < min_dist
			min_dist = dist
			closest = i
		end
	end

	return closest

end

function generate_accel(bmodel::AvoidModel, dmodel::IDMMOBILModel, s::MLState, neighborhood::Array{Int,1}, idx::Int, rng::AbstractRNG)
	closest_car = closest_car(dmodel,s,neighborhood,idx,bmodel.jerk)
	if closest_car == 0
		return 0.
	end

	x = s.env_cars[idx].pos[1]
	cc = s.env_cars[closest_car]

	if p.jerk #TODO fix this so it doesn't always accelerate if there's no space
    accel = cs.pos[1] - x > 2*dmodel.phys_param.l_car ? 1 : -3. #slam brakes
  else
    accel = -sign(cs.pos[1] - x)
  end

	return accel*dmodel.phys_param.dt #XXX times accel interval?

end

function generate_lane_change(bmodel::AvoidModel, dmodel::IDMMOBILModel, s::MLState, neighborhood::Array{Int,1}, idx::Int, rng::AbstractRNG)
	closest_car = closest_car(dmodel,s,neighborhood,idx,bmodel.jerk)
	if closest_car == 0
		return 0.
	end

	pos = s.env_cars[idx].pos[2]
	cc = s.env_cars[closest_car]

	lanechange = -sign(cc.pos[2] - pos)
	#if same lane, go in direction with more room
	if lanechange == 0
		if pos >= 2*dmodel.phys_param.nb_lanes - 1 - pos >= pos #more room to left >= biases to left
			lanechange = 1
		else
			lanechange = -1
		end
	end
	if lanechange > 0 && pos >= 2*dmodel.phys_param.nb_lanes - 1
		lanechange = 0
	elseif lanechange < 0 && pos <= 1
		lanechange = 0
	end

	return lanechange
end
