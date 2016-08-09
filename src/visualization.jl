# using PyPlot
using Reel

BEHAVIOR_COLORS = Dict{Float64,AbstractString}(0.5=>"#0101DF",0.25=>"#D7DF01",0.0=>"#FF0000")

draw_box(x::Union{Float64,Int},y::Union{Float64,Int},w::Union{Float64,Int},h::Union{Float64,Int})=fill([x;x+w;x+w;x],[y;y;y+h;y+h])
draw_box(x::Union{Float64,Int},y::Union{Float64,Int},w::Union{Float64,Int},h::Union{Float64,Int},color::AbstractString)=fill([x;x+w;x+w;x],[y;y;y+h;y+h],color=color)


function display_sim(mdp, S::Array{MLState,1}, A::Array{MLAction,1})
    warn("This should be run in a Jupyter Notebook")
    assert(length(S) == length(A)+1)
    @manipulate for i = 1:length(A)
        visualize(mdp,S[i],A[i],S[i+1])
    end
end

function display_sim(mdp, sim::HistoryRecorder)
    display_sim(mdp, sim.state_hist, sim.action_hist)
end

function write_tmp_gif(mdp, sim::HistoryRecorder)
    dt = mdp.dmodel.phys_param.dt
    S = sim.state_hist
    A = sim.action_hist
    film = roll(fps = 1/dt, duration = dt*(length(sim.action_hist)-1)) do t, dt
        i = round(Int, t/dt)+1
        print('.')
        visualize(mdp, S[i], A[i], S[i+1])
    end
    filename = string(tempname(), ".gif")
    write(filename, film)
    return filename
end

function visualize(mdp::Union{MLMDP,MLPOMDP},
                   s::MLState, a::MLAction, sp=create_state(mdp))
    visualize(mdp, s)
end

function visualize(mdp::Union{MLMDP,MLPOMDP}, s::MLState)
    pp = mdp.dmodel.phys_param
    roadway = gen_straight_roadway(pp.nb_lanes,
                                   pp.lane_length,
                                   lane_width=pp.w_lane)
    scene = Scene()
    for cs in s.env_cars
        push!(scene, Vehicle(VehicleState(VecSE2(cs.x, (cs.y-1.0)*pp.w_lane, 0.0), roadway, 30.0), 
                                VehicleDef(cs.id, AgentClass.CAR, pp.l_car, pp.w_car)))
    end
    render(scene, roadway, cam=FitToContentCamera())
end

#=
function draw_bang(x::Union{Float64,Int},
										y::Union{Float64,Int},
										color::AbstractString="#FFBF00";
    								a::Union{Float64,Int}=2.,
										b::Union{Float64,Int}=a,
    								nb_spikes::Int=8,
										spike_depth::Union{Float64,Int}=1.5,
										th_offset::Union{Float64,Int}=0.,
    								th_irregularity::Float64=0.5,
										r_irregularity::Float64=0.5,
										rng::AbstractRNG=MersenneTwister(1),
    								edge_color::AbstractString="k")
    #A somewhat overly complicated function for drawing the crash graphic
    X = zeros(nb_spikes*2)
    Y = zeros(nb_spikes*2)

    th_interval = 2.*pi/length(X)

    for i = 1:length(X)
        if i % 2 == 1
            a_ = a+spike_depth*(0.5 + r_irregularity*(rand(rng)-0.5))
            b_ = b+spike_depth*(0.5 + r_irregularity*(rand(rng)-0.5))
        else
            a_ = a-spike_depth*(0.5 + r_irregularity*(rand(rng)-0.5))
            b_ = b-spike_depth*(0.5 + r_irregularity*(rand(rng)-0.5))
        end
        th = (i+th_irregularity*(rand(rng)-0.5))*th_interval+th_offset
        X[i] = x + a_*cos(th)
        Y[i] = y + b_*sin(th)
    end

    fill(X,Y,color=color,ec=edge_color)
end
=#

#=
function draw_direction(phys_param::PhysicalParam, x::Float64, y::Float64, v_nom::Float64, s::CarState)
	#plot desired car speed relative to self car as an opaque arrow
	#arrow should be at most one car in length, and ~1.2 of a car width at headway
	#NOTE: the arrow head is extra length
	max_arrow_length = 1.125*phys_param.l_car
	phantom_arrow_length = max_arrow_length*(s.vel-v_nom)/(phys_param.v_fast-phys_param.v_slow)
	phantom_arrow_head_width = 1.2*phys_param.w_car
	phantom_arrow_body_width  = 0.8*phys_param.w_car
	phantom_arrow_head_length = phantom_arrow_length/3.
	arrow(x,y,phantom_arrow_length,0.,head_width=phantom_arrow_head_width,
				head_length=phantom_arrow_head_width,width=phantom_arrow_body_width,
				alpha=0.5,color="#1C1C1C")

	#draw actual speed/direction; this is where the center of the car will be in
	#the next time step, approximately

	dx = phys_param.dt*(s.vel-v_nom)
    dy = s.lane_change*phys_param.w_lane
	hw = 0.5*phys_param.w_car
	hl = 1.5*hw
	w = 0.75*hw
	th = atan2(dy,dx)
	dx -= hl*cos(th)
	dy -= hl*sin(th)
	arrow(x,y,dx,dy,width=w,head_width=hw,head_length=hl,fc="#DF7401", ec="#0404B4",alpha=0.75)
	#terrible looking high contrast blue/orange arrows

end

function draw_sedan(pp::PhysicalParam, s::CarState, v_nom::Float64, frame::Float64=0., INTERVAL::Float64=0.)
	if s.x < 0.
		#oob
		return
	end
	lane_length = pp.lane_length
	x_ctr = s.x+frame
	y_ctr = pp.w_lane*s.y - INTERVAL*floor(Integer,x_ctr/lane_length)*(frame == 0. ? 0.: 1.)
	x_ctr = mod(x_ctr,lane_length+0.0001)
  #Fix nullable TODO
  if isnull(s.behavior)
    p = -1.
  else
    p = get(s.behavior).p_mobil.p
   end
   color = get(BEHAVIOR_COLORS,p,"#B404AE") #PLACEHOLDER
	draw_direction(pp,x_ctr,y_ctr,v_nom,s)
	draw_sedan(pp,x_ctr,y_ctr,color)
	annotate("$(round(s.vel,2))",xy=(x_ctr,y_ctr))
    annotate(string(s.id), xy=(x_ctr, y_ctr-2))
end
=#
