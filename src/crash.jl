#crash.jl
#a separate file for all the crashing stuff

function cross2d(x::Array{Float64,1},y::Array{Float64,1})
	assert(length(x) == length(y))
	if length(x) == 3
		return cross(x,y)
	elseif length(x) != 2
		error("Incorrect length for cross product")
	end

	return x[1]*y[2]-x[2]*y[1]
end

function line_segment_intersect(p::Array{Float64,1},pr::Array{Float64,1},q::Array{Float64,1},qs::Array{Float64,1})
	if (length(p) != length(q)) || (length(p) != length(pr)) || (length(q) != length(qs))
		error("Error: inconsistent dimensionality")
	end

	s = qs - q
	r = pr - p

	rxs = cross2d(r,s)
	q_pxr = cross2d(q-p,r)
	if rxs == 0
		if q_pxr == 0
			#collinear
			t0 = dot(q-p,r)/dot(r,r)
			t1 = t0 + dot(s,r)/dot(r,r)
			if dot(r,s) > 0
				if max(0.,min(t1,1.)-max(0.,t0)) > 0
					return true
				end
				return false
			else
				if max(0.,min(t0,1.)-max(0.,t1)) > 0
					return true
				end
				return false
			end
		else
			return false
		end
	end

	t = cross2d(q-p,s)/cross2d(r,s)
	u = cross2d(q-p,r)/cross2d(r,s)

	if (max(0.,min(1.,t)) == t) && (max(0.,min(1.,u)) == u)
		return true
	end

	return false
end

function poly_intersect(X::Array{Array{Float64,2},1},Y::Array{Array{Float64,2},1})
	#each row corresponds to x, y, z....
	#need i,i+1 to be each line segment
	#does pairwise comparison of line segments in O(n2) time
	##TODO: still need to check the cases of being completely inside of one another

	for x in X
		for y in Y
			if size(x) != size(y)
				error("Error: ill formed input1")
			end
			if size(x)[2] > 2
				error("Error: ill formed input2")
			end
			# This is correct: input has format: [x x'; y y']
			#  so each slice is [x;y] and [x';y'] respectively
			if line_segment_intersect(x[:,1],x[:,2],y[:,1],y[:,2]) # XXX this doesnt seem right
				return true
			end
		end
	end

	#check if one is inside the other
	max1 = max(X[1][:,1],X[1][:,2])
	min1 = min(X[1][:,1],X[1][:,2])
	max2 = max(Y[1][:,1],Y[1][:,2])
	min2 = min(Y[1][:,1],Y[1][:,2])
	for i = 2:length(X)
		max1 = max(max1,max(X[i][:,1],X[i][:,2]))
		min1 = min(min1,min(X[i][:,1],X[i][:,2]))
	end
	for i = 2:length(Y)
		max2 = max(max2,max(Y[i][:,1],Y[i][:,2]))
		min2 = min(min2,min(Y[i][:,1],Y[i][:,2]))
	end

	#Case: X is contained in Y
	if !(false in (max1 .< max2)) && !(false in (min1 .> min2))
		return true
	end
	#Case: Y is contained in X
	if !(false in (max2 .< max1)) && !(false in (min2 .> min1))
		return true
	end

	return false
end

function is_crash(mdp::MLMDP{MLState,MLAction},s::MLState,a::MLAction,debug::Bool=false)
	#calculate current position, next position, convert to metric space
	#convert to polyhedron based on car size
	#do collision check between agent car and all environment cars
	#going offroad is considered grashing

    pp = mdp.dmodel.phys_param
    nb_col = 2*pp.nb_lanes-1
	agent_pos = pp.lane_length/2.
	agent_y = s.agent_pos*pp.y_interval

	#treat agent_pos, agent_y as (0,0)
	w_car = pp.w_car
	l_car = pp.l_car
	#TODO: make it so that X takes in to account the fact that the agent car can change lanes
	w_car_ = w_car
	diff = 0.75
	if a.lane_change < 0
		agent_y -= pp.y_interval
		w_car_ += pp.y_interval*diff
	elseif a.lane_change > 0
		w_car_ += pp.y_interval*(1.+(1-diff))
		agent_y += pp.y_interval*(1-diff)

	end
	#X = Array{Float64,2}[agent_pos agent_pos+l_car agent_pos+l_car agent_pos agent_pos; agent_y agent_y agent_y+w_car agent_y+w_car agent_y]
	X = Array{Float64,2}[[agent_pos agent_pos; agent_y agent_y+w_car_],[agent_pos+l_car agent_pos+l_car; agent_y agent_y+w_car_],
						[agent_pos agent_pos+l_car; agent_y agent_y],[agent_pos agent_pos+l_car; agent_y+w_car_ agent_y+w_car_]]

	#center of the ego car
	x = [agent_pos; agent_y]
	#corner-corner distance between two sedan
	d = norm([2*l_car;2*w_car])
	if debug
		subplot(212)
		for x in X
			plot(vec(x[1,:]),vec(x[2,:]),color="k")
		end
	end

	dt = pp.dt
	for (i,env_car) in enumerate(s.env_cars)
		pos = env_car.pos
		if pos[1] < 0.
			continue
		end
		vel = env_car.vel
		lane_change = env_car.lane_change
		behavior = env_car.behavior
		lane_ = max(1,min(pos[2]+lane_change,nb_col))
		neighborhood = get_adj_cars(pp,s.env_cars,i)

		dv = get(neighborhood.ahead_dv,0,0.)
		ds = get(neighborhood.ahead_dist,0,1000.)

        # TODO first do a quick check to see if the cars are even close

		dvel_ms = get_idm_dv(behavior.p_idm,dt,vel,dv,ds) #call idm model
		dp =  dt*(vel-s.agent_vel)#dt*(pp.VELOCITIES[vel]-pp.VELOCITIES[s.agent_vel])#+0.5*dt*dvel_ms #x+vt+1/2at2 #XXX remove at2 term
		dy = (lane_-pos[2])*pp.y_interval # XXX this doesn't seem right
		#dy = lane_change*pp.y_interval
		p = pos[1]#pp.POSITIONS[pos[1]]
		y = pos[2]*pp.y_interval

		y1 = [p;y]
		y2 = [p+dp;y+dy]
		if norm(x-y1) > d && norm(x-y2) > d
			continue
		end
		##TODO: make consistent with new formulation
		Y1 = Array{Float64,2}[[p p; y y+w_car],[p+l_car p+l_car; y y+w_car],[p p+l_car; y y],[p p+l_car; y+w_car y+w_car]]
		Y2 = Array{Float64,2}[xy + [dp dp; dy dy] for xy in Y1]
		Y3 = Array{Float64,2}[[p p+dp; y y+dy],[p+l_car p+dp+l_car; y y+dy],[p p+dp; y+w_car y+dy+w_car],[p+l_car p+dp+l_car; y+w_car y+dy+w_car]]

		Y = Array{Float64,2}[Y1;Y2;Y3]
		if debug
			for Yi in Y
				plot(vec(Yi[1,:]),vec(Yi[2,:]),color="b")
			end
		end
		if poly_intersect(X,Y)
			return true
		end
	end

	return false
end
