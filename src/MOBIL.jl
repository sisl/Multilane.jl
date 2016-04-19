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

type CarNeighborhood
	ahead_dist::Dict{Int,Float64} #(-1,0,1) for right,self, and left lane
	behind_dist::Dict{Int,Float64}
	ahead_dv::Dict{Int,Float64}
	behind_dv::Dict{Int,Float64}
	ahead_idm::Dict{Int,IDMParam}
	behind_idm::Dict{Int,IDMParam}
	ahead_lanechange::Int
end
CarNeighborhood(x,y,z) = CarNeighborhood(x,deepcopy(x),deepcopy(x),deepcopy(x),y,deepcopy(y),z)
CarNeighborhood() = CarNeighborhood(Dict{Int,Float64}(),Dict{Int,IDMParam}(),0)

function is_lanechange_dangerous(nbhd::CarNeighborhood,dt::Float64,l_car::Float64,dir::Int)
	slf = get(nbhd.behind_dist,dir,1000.)
	slb = get(nbhd.ahead_dist,dir,1000.)
	dvlf = get(nbhd.behind_dv,dir,0.)
	dvlb = get(nbhd.ahead_dv,dir,0.)
	dslf = slf-dvlf*dt
	dslb = slb-dvlb*dt
	diff = 0.0 #something >=0--the safety distance
	#println(slf)
	#println(slb)
	#println(dslf)
	#println(dslb)

	return (slf < diff*l_car) || (slb < diff*l_car) || dslb < diff*l_car ||
				dslf < diff*l_car

end

function get_adj_cars(p::PhysicalParam,arr::Array{CarState,1},i::Int)
	##TODO: update CarNeighborhood with stuff for the IDM parameters
	neighborhood = CarNeighborhood()

	x = arr[i]

	#going offroad is not allowed
	if x.pos[2] <= 1
		#can't go right
		neighborhood.ahead_dist[-1] = -100.
		neighborhood.behind_dist[-1] = -100.
	# elseif x.pos[2] >= p.NB_POS/length(p.POSITIONS)
    # XXX is the line below the same as the line above??
	elseif x.pos[2] >= 2*p.nb_lanes-1
		#can't go left
		neighborhood.ahead_dist[1] = -100.
		neighborhood.behind_dist[1] = -100.
	end

	for (j,car) in enumerate(arr)
		if i == j
			continue
		end
		pos = car.pos
		vel = car.vel
		if car.pos[1] < 0.
			continue
		end
		dlane = pos[2]-x.pos[2]
		if abs(dlane) > 2 #not in or adjacent to current lane
			continue
		elseif abs(dlane) <= 1
			dlane = 0
		else #abs(dlane) == 2
			dlane = sign(dlane)
		end
		#if abs(dlane) == 1 or 0, consider to be in same lane; 2 next lane, more, ignore

		dist = pos[1]-x.pos[1]#p.POSITIONS[pos[1]]-p.POSITIONS[x.pos[1]]
		dv = x.vel-vel#p.VELOCITIES[x.vel]-p.VELOCITIES[vel]

		if (dist >= 0) && ((dist - p.l_car) < get(neighborhood.ahead_dist,dlane,1000.))
			neighborhood.ahead_dist[dlane] = dist - p.l_car
			neighborhood.ahead_dv[dlane] = dv
			neighborhood.ahead_idm[dlane] = car.behavior.p_idm #pointless
			if dlane == 0
				neighborhood.ahead_lanechange = car.lane_change
			end
		elseif (dist < 0) && ((-1*dist - p.l_car) < get(neighborhood.behind_dist,dlane,1000.))
			neighborhood.behind_dist[dlane] = -1*dist - p.l_car
			neighborhood.behind_dv[dlane] = -1*dv
			neighborhood.behind_idm[dlane] = car.behavior.p_idm
		end
	end

	return neighborhood #PLACEHOLDER

end

function get_mobil_lane_change(p::PhysicalParam,state::CarState,neighborhood::CarNeighborhood)
	#TODO: catch if the parameters don't exist

	#need 6 distances: distance to person behind me, ahead of me
	#				potential distance to person behind me, ahead of me
	#				in other lane(s)
	#need sets of idm parameters
	dt = p.dt
	p_idm_self = state.behavior.p_idm
	p_mobil = state.behavior.p_mobil
	#println(neighborhood)
	if isempty(neighborhood.ahead_dv) && isempty(neighborhood.behind_dv)
		return 0 #no reason to change lanes if you're all alone
	end

	##if between lanes, return +1 if moving left, -1 if moving right
	if state.pos[2] % 2 == 0 #even is between lanes
		return state.lane_change #continue going in the direction you're going
	end

	v = state.vel#p.VELOCITIES[state.vel]
	#get_idm_dv(param,velocity,dv,s)
	#get predicted and potential accelerations
	a_self = get_idm_dv(p_idm_self,dt,v,get(neighborhood.ahead_dv,0,0.),get(neighborhood.ahead_dist,0,1000.))/dt
	a_self_left = get_idm_dv(p_idm_self,dt,v,get(neighborhood.ahead_dv,1,0.),get(neighborhood.ahead_dist,1,1000.))/dt
	a_self_right = get_idm_dv(p_idm_self,dt,v,get(neighborhood.ahead_dv,-1,0.),get(neighborhood.ahead_dist,-1,1000.))/dt

	if get(neighborhood.behind_idm,0,0.) == 0.
		a_follower = 0.
		a_follower_ = 0.
	else
		v_behind = v + neighborhood.behind_dv[0]
		dv_behind = neighborhood.behind_dv[0]
		s_behind = get(neighborhood.behind_dist,0,1000.)
		a_follower = get_idm_dv(neighborhood.behind_idm[0],dt,v_behind,dv_behind,s_behind)/dt #distance behind is a negative number
		dv_behind_ = dv_behind + get(neighborhood.ahead_dv,0,0.)
		s_behind_ = s_behind + get(neighborhood.ahead_dist,0,1000.) + p.l_car
		a_follower_ = get_idm_dv(neighborhood.behind_idm[0],dt,v_behind,dv_behind_,s_behind_)/dt
	end

	if get(neighborhood.behind_idm,1,0.) == 0.
		a_follower_left = 0.
		a_follower_left_ = 0.
	else
		v_left = v + neighborhood.behind_dv[1]
		dv_left = neighborhood.behind_dv[1] + get(neighborhood.ahead_dv,1,0.)
		s_left = get(neighborhood.behind_dist,1,1000.) + get(neighborhood.ahead_dist,1,1000.)+p.l_car
		a_follower_left = get_idm_dv(neighborhood.behind_idm[1],dt,v_left,dv_left,s_left)/dt
		dv_left_ = get(neighborhood.behind_dv,1,0.)
		s_left_ = get(neighborhood.behind_dist,1,1000.)
		a_follower_left_ = get_idm_dv(neighborhood.behind_idm[1],dt,v_left,dv_left_,s_left_)/dt
	end

	if get(neighborhood.behind_idm,-1,0.) == 0.
		a_follower_right = 0.
		a_follower_right_ = 0.
	else
		v_right = v + neighborhood.behind_dv[-1]
		dv_right = neighborhood.behind_dv[-1] + get(neighborhood.ahead_dv,-1,0.)
		s_right = neighborhood.behind_dist[-1] + get(neighborhood.ahead_dist,-1,1000.) + p.l_car
		a_follower_right = get_idm_dv(neighborhood.behind_idm[-1],dt,v_right,dv_right,s_right)/dt
		dv_right_ = neighborhood.behind_dv[-1]
		s_right_ = neighborhood.behind_dist[-1]
		a_follower_right_ = get_idm_dv(neighborhood.behind_idm[-1],dt,v_right,dv_right_,s_right_)/dt
	end


	#calculate incentives
	left_crit = a_self_left-a_self+p_mobil.p*(a_follower_left_-a_follower_left+a_follower_-a_follower)
	right_crit = a_self_right-a_self+p_mobil.p*(a_follower_right_-a_follower_right+a_follower_-a_follower)


	#check safety criterion, also check if there is physically space
	if (a_follower_right_ < -p_mobil.b_safe) && (a_follower_left_ < -p_mobil.b_safe)
		return 0 #neither safe
	end
	if is_lanechange_dangerous(neighborhood,dt,p.l_car,1) || (a_follower_left_ < -p_mobil.b_safe)
		left_crit -= 10000000.
	end

	if is_lanechange_dangerous(neighborhood,dt,p.l_car,-1) || (a_follower_right_ < -p_mobil.b_safe)
		right_crit -= 10000000.
	end

	#println(neighborhood)
	#r = state.behavior.rationality
	#v0 = p_idm_self.v0
	#println("$(r) $(v0) left: $left_crit,$slf,$slb,$dslf,$dslb\n right: $right_crit,$slf_,$slb_,$dslf_,$dslb_")

	#check if going left or right is preferable
	dir_flag = left_crit >= right_crit ? 1:-1
	#check incentive criterion
	if max(left_crit,right_crit) > p_mobil.a_thr
		return dir_flag
	end
	return 0
end
