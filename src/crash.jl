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

# to see, run visualize(mdp, s, a, sp, two_frame_crash=true, debug=true)
function is_crash(mdp::Union{MLMDP{MLState,MLAction},MLPOMDP{MLState,MLAction}}, s::MLState, sp::MLState, debug::Bool=false; warning::Bool=false)
	pp = mdp.dmodel.phys_param
	dt = pp.dt
  nb_col = 2*pp.nb_lanes-1
	agent_pos = s.env_cars[1].x#pp.lane_length/2.
	agent_y = s.env_cars[1].y*pp.w_lane
	agent_vel = s.env_cars[1].vel

	#treat agent_pos, agent_y as (0,0)
	w_car = pp.w_car
	l_car = pp.l_car
	#TODO: make it so that X takes in to account the fact that the agent car can change lanes
	w_car_ = w_car

	agent_y_ = sp.env_cars[1].y*pp.w_lane
	if agent_y_ <= agent_y
		w_car_ += agent_y - agent_y_
		agent_y = agent_y_
	else
		w_car_ += agent_y_ - agent_y
	end
	X = Array{Float64,2}[[agent_pos agent_pos; agent_y agent_y+w_car_],[agent_pos+l_car agent_pos+l_car; agent_y agent_y+w_car_],
						[agent_pos agent_pos+l_car; agent_y agent_y],[agent_pos agent_pos+l_car; agent_y+w_car_ agent_y+w_car_]]

	#center of the ego car
	x = [agent_pos; agent_y]
	x2 = [sp.env_cars[1].x; agent_y_]
	#corner-corner distance between two sedan
	d = norm([2*l_car;2*w_car])
	if debug
		subplot(212)
		for _x in X
			plot(vec(_x[1,:]),vec(_x[2,:]),color="k")
		end
	end

	#another way to ignore ego car
	id = Dict{Int,Int}([car.id=>i+1 for (i,car) in enumerate(s.env_cars[2:end])])
	idp = Dict{Int,Int}([car.id=>i+1 for (i,car) in enumerate(sp.env_cars[2:end])])
	ids = intersect(keys(id), keys(idp))
	# NOTE: VVV
	#if its not in both, then its entering or leaving and thus at extremum of track
	#and most likely not eligible for crash detection

	cars = Tuple{CarState,CarState}[(s.env_cars[id[car_id]],
                                            sp.env_cars[idp[car_id]],)
										    for car_id in ids] #1 or 2

	for (env_car,env_car_) in cars

		if env_car.id == 1 #presumably ego car XXX probably not needed
			continue
		end
		if env_car.x < 0. # XXX probably unneeded
			continue
		end
		p = env_car.x
		y = env_car.y*pp.w_lane
		p_ = env_car_.x
		yp = env_car_.y * pp.w_lane

		dp = p_ - p
		dy = yp - y

		y1 = [p;y]
		y2 = [p_;yp]
		if norm(x-y1) > d && norm(x-y2) > d && norm(x2-y1) > d && norm(x2-y2) > d
			continue
		end
		##TODO: make consistent with new formulation
		Y1 = Array{Float64,2}[[p p; y y+w_car],[p+l_car p+l_car; y y+w_car],[p p+l_car; y y],[p p+l_car; y+w_car y+w_car]]
		Y2 = Array{Float64,2}[xy + [dp dp; dy dy] for xy in Y1]
		Y3 = Array{Float64,2}[[p p_; y yp],[p+l_car p_+l_car; y yp],[p p_; y+w_car yp+w_car],[p+l_car p_+l_car; y+w_car yp+w_car]]

		Y = Array{Float64,2}[Y1;Y2;Y3]
		if debug
			for Yi in Y
				plot(vec(Yi[1,:]),vec(Yi[2,:]),color="b")
			end
		end
		if poly_intersect(X,Y)
            if warning
                crash_warning(mdp, s, sp)
            end
			return true
		end
	end

	return false
end

function crash_warning(mdp, s, sp)
    warn("""
    Crash!

    mdp = $mdp

    s = $s

    sp = $sp
    """)
end
