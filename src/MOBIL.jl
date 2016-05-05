#MOBIL.jl
#A separate file for all the MOBIL stuff

###############
##MOBIL Model##
###############

immutable MOBILParam
	p::Float64 #politeness factor
	b_safe::Float64 #safe braking value
	a_thr::Float64 #minimum accel
	#db::Float64 #lane bias #we follow the symmetric/USA lane change rule
end #MOBILParam
MOBILParam(;p::Float64=0.25,b_safe::Float64=4.,a_thr::Float64=0.2) = MOBILParam(p,b_safe,a_thr)
==(a::MOBILParam,b::MOBILParam) = (a.p==b.p) && (a.b_safe==b.b_safe) &&(a.a_thr == b.a_thr)
Base.hash(a::MOBILParam,h::UInt64=zero(UInt64)) = hash(a.p,hash(a.b_safe,hash(a.a_thr,h)))

function MOBILParam(s::AbstractString)
	#typical politeness range: [0.0,0.5]
	typedict = Dict{AbstractString,Float64}("cautious"=>0.5,"normal"=>0.25,"aggressive"=>0.0)
	p = get(typedict,s,-1.)
	assert(p >= 0.)
	return MOBILParam(p=p)
end

function is_lanechange_dangerous(pp::PhysicalParam,s::MLState,nbhd::Array{Int,1},idx::Int,dir::Real)

	#check if dir is oob
	lane_ = s.env_cars[idx].y + dir
	if (lane_ > pp.nb_lanes) || (lane_ < 1.)
		return true
	end
	#check if will hit car next to you?
	dt = pp.dt
	l_car = pp.l_car

	dvlf, slf = get_dv_ds(pp,s,nbhd,idx,round(Int, 5+dir))
	dvlb, slb = get_dv_ds(pp,s,nbhd,idx,round(Int, 2+dir))

	#distance at next time step
	#dv ref: ahead: me - him; behind: him - me
	#ds ref: ahead: him - me - l_car; behind: me - him - l_car
	dslf = slf-dvlf*dt
	dslb = slb-dvlb*dt
	diff = 0.0 #something >=0--the safety distance

	return (slf < diff*l_car) || (slb < diff*l_car) || dslb < diff*l_car ||
				dslf < diff*l_car

end

function get_neighborhood(pp::PhysicalParam,s::MLState,idx::Int)
	nbhd = zeros(Int,6)
	dists = Inf*ones(6)
	"""
	Each index corresponds to the index (in the car state) that corresponds to
	the position, 0 if there is no such car
	indices:
	6      |     3
	5    |car|   2 ->
	4      |     1
	where 1/2/3 is the front, 3/6 is left
	"""
	x = s.env_cars[idx]
	#rightmost lane: no one to the right
	if x.y <= 1.
		dists[[1;4]] = [-1.;-1.]
	#leftmost lane: no one to the left
	elseif x.y >= pp.nb_lanes
		dists[[3;6]] = [-1.;-1.]
	end

	for (i,car) in enumerate(s.env_cars)
		#i am not a neighbor
		if i == idx
			continue
		end
		pos = car.x
		lane = car.y
		#this car is oob
		if pos < 0.
			continue
		end
		dlane = lane - x.y #NOTE float: convert to int
		#too distant to be a neighbor
		if abs(dlane) > 1.
			continue
		elseif abs(dlane) <= 0.5
			dlane = 0
		else
			dlane = convert(Int,sign(dlane))
		end

		d = pos-x.x

		offset = d >= 0. ? 2 : 5

		if abs(d) < dists[offset+dlane]
			dists[offset+dlane] = abs(d)
			nbhd[offset+dlane] = i
		end
	end

	return nbhd
end

function get_dv_ds(pp::PhysicalParam,s::MLState,nbhd::Array{Int,1},idx::Int,idy::Int)
	"""
	idx is the car from whom the perspective is
	idy is the other car
	"""
	car = s.env_cars[idx]
	nbr = nbhd[idy]
	#dv: if ahead: me - him; behind: him - me
	dv = nbr != 0 ? -1*sign((idy-3.5))*(car.vel - s.env_cars[nbr].vel) : 0.
	ds = nbr != 0 ? abs(s.env_cars[nbr].x - car.x) - pp.l_car : 1000.

	return dv, ds
end

function get_rear_accel(pp::PhysicalParam,s::MLState,nbhd::Array{Int,1},idx::Int,dir::Int)

	#there is no car behind in that spot
	if nbhd[5+dir] == 0
		return 0., 0.
	end

	if isnull(s.env_cars[nbhd[5+dir]].behavior)
		return 0., 0.
	end
	if !(typeof(get(s.env_cars[nbhd[5+dir]].behavior)) <: IDMMOBILBehavior)
		#TODO figure out
		# need some kind of api to interface with other behavior model types
		# for now: just assume 0, 0 (no effect if they're not IDM)
		a = 0.
		return a, a
	end
	v = s.env_cars[idx].vel

	#behind - me
	dv_behind, s_behind = get_dv_ds(pp,s,nbhd,idx,5+dir)
	v_behind = v - dv_behind

	#me - front
	#what would the relative velocity, distance be if idx wasn't there
	dv_behind_, s_behind_ = get_dv_ds(pp,s,nbhd,idx,2+dir)
	dv_behind_ += dv_behind
	s_behind_ += s_behind + pp.l_car

	dt = pp.dt
	#TODO generalize to get_dv?
	behind_idm = get(s.env_cars[nbhd[5+dir]].behavior).p_idm
	a_follower = get_idm_dv(behind_idm,dt,v_behind,dv_behind,s_behind)/dt #distance behind is a negative number
	a_follower_ = get_idm_dv(behind_idm,dt,v_behind,dv_behind_,s_behind_)/dt

	return a_follower, a_follower_
end

function get_mobil_lane_change(pp::PhysicalParam,s::MLState,nbhd::Array{Int,1},idx::Int,rng::AbstractRNG=MersenneTwister(123))
	#TODO: catch if the parameters don't exist

	#need 6 distances: distance to person behind me, ahead of me
	#				potential distance to person behind me, ahead of me
	#				in other lane(s)
	#need sets of idm parameters
	dt = pp.dt
	state = s.env_cars[idx]
	p_idm_self = get(state.behavior).p_idm
	p_mobil = get(state.behavior).p_mobil
	#println(neighborhood)
	if sum(nbhd) == 0
		return 0 #no reason to change lanes if you're all alone
	end

	##if between lanes, return +1 if moving left, -1 if moving right
	if state.y-0.5 % 1 == 0 #even is between lanes
		return state.lane_change #continue going in the direction you're going
	end

	v = state.vel
	#get predicted and potential accelerations

	#TODO generalize to get_dv()?
	a_self = get_idm_dv(p_idm_self,dt,v,get_dv_ds(pp,s,nbhd,idx,2)...)/dt
	a_self_left = get_idm_dv(p_idm_self,dt,v,get_dv_ds(pp,s,nbhd,idx,3)...)/dt
	a_self_right = get_idm_dv(p_idm_self,dt,v,get_dv_ds(pp,s,nbhd,idx,1)...)/dt

	a_follower, a_follower_ = get_rear_accel(pp,s,nbhd,idx,0)
	a_follower_left_, a_follower_left = get_rear_accel(pp,s,nbhd,idx,1)
	a_follower_right_, a_follower_right = get_rear_accel(pp,s,nbhd,idx,-1)

	#calculate incentives
	left_crit = a_self_left-a_self+p_mobil.p*(a_follower_left_-a_follower_left+a_follower_-a_follower)
	right_crit = a_self_right-a_self+p_mobil.p*(a_follower_right_-a_follower_right+a_follower_-a_follower)

	#check safety criterion, also check if there is physically space
	if (a_follower_right_ < -p_mobil.b_safe) && (a_follower_left_ < -p_mobil.b_safe)
		return 0 #neither safe
	end
	if is_lanechange_dangerous(pp,s,nbhd,idx,1) || (a_follower_left_ < -p_mobil.b_safe)
		left_crit -= 10000000.
	end

	if is_lanechange_dangerous(pp,s,nbhd,idx,-1) || (a_follower_right_ < -p_mobil.b_safe)
		right_crit -= 10000000.
	end
	#check if going left or right is preferable
	dir_flag = left_crit >= right_crit ? 1.:-1.
	#check incentive criterion
	if max(left_crit,right_crit) > p_mobil.a_thr
		return dir_flag
	end
	return 0.
end
