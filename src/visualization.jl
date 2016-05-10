using PyPlot

BEHAVIOR_COLORS = Dict{Float64,AbstractString}(0.5=>"#0101DF",0.25=>"#D7DF01",0.0=>"#FF0000")

draw_box(x::Union{Float64,Int},y::Union{Float64,Int},w::Union{Float64,Int},h::Union{Float64,Int})=fill([x;x+w;x+w;x],[y;y;y+h;y+h])
draw_box(x::Union{Float64,Int},y::Union{Float64,Int},w::Union{Float64,Int},h::Union{Float64,Int},color::AbstractString)=fill([x;x+w;x+w;x],[y;y;y+h;y+h],color=color)


function display_sim(mdp, S::Array{MLState,1}, A::Array{MLAction,1}; debug::Bool=false)
  warn("This should be run in a Jupyter Notebook")
  assert(length(S) >= length(A))
  f = figure()
  @manipulate for i = 1:length(A); withfig(f) do
    visualize(mdp,S[i],A[i],debug=debug) end
  end
end

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

function draw_sedan(phys_param::PhysicalParam, x_ctr::Float64, y_ctr::Float64, color::AbstractString)
	w = phys_param.l_car
	h = phys_param.w_car
	x = x_ctr - 0.5*w
	y = y_ctr - 0.5*h

	##TODO: color stuff based on behavior Model
	draw_box(x,y,w,h,color)
	window_color = "#848484"
	x_windshield = x_ctr
	w_windshield = w*0.25
	h_windshield = h*0.75
	y_windshield = y_ctr - 0.5*h_windshield
	draw_box(x_windshield,y_windshield,w_windshield,h_windshield,window_color)
	x_backwindow = x_ctr - 0.375*w
	w_backwindow = w*0.125
	h_backwindow = h*0.75
	y_backwindow = y_ctr - 0.5*h_backwindow
	draw_box(x_backwindow,y_backwindow,w_backwindow,h_backwindow,window_color)
	##TODO: add direction arrow
	##TODO: add a face based on behavior model
end

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
    dy = s.lane_change*phys_param.y_interval
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
	y_ctr = pp.y_interval*s.y - INTERVAL*floor(Integer,x_ctr/lane_length)*(frame == 0. ? 0.: 1.)
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
end

function visualize(mdp::MLMDP,
                   s::MLState, a::MLAction, sp=create_state(mdp); debug::Bool=false, frame::Float64=0., nb_rows::Int=1, two_frame_crash::Bool=false)
	#Placeholder!
	clf()
	if debug
		subplot(211)
	end

    pp = mdp.dmodel.phys_param
    nb_col = 2*pp.nb_lanes-1

	#println("a")
	lane_length = pp.lane_length
	#TODO functionize this, have it work over multiple rows
	#make the initial canvas
	W = pp.lane_length
	H = pp.w_lane*(nb_col+1)/2
	border_thickness = 2.
	border_color = "#000000" #black
	#draw the lane markings 3ft long, 9ft btwn (bikes), 10cm wide
	#~1m x 0.1m for an approximation
	lane_marker_length = 1.
	lane_marker_width = 0.1
	lane_marker_inbetween = 4. #1+3
	ROW_INTERVAL = H+2*border_thickness
	#println("b")

	for row = 0:(nb_rows-1)
		#road
		Y_ABS = -1*row*ROW_INTERVAL + 2. #XXX hack
		draw_box(0,Y_ABS,W,H,"#424242") #a gray

		#draw hard borders on top and bottom
		draw_box(0,-border_thickness+Y_ABS,W,border_thickness,border_color)
		draw_box(0.,H+Y_ABS,W,border_thickness,border_color)

        # \/ rand is there to make it look like the cars are moving
		lane_marker_x = collect((0.-rand()):lane_marker_inbetween:W)
		lane_marker_length_ = Float64[lane_marker_length for i=1:length(lane_marker_x)]
		lane_marker_length_[1] += lane_marker_x[1]
		lane_marker_length_[end] = lane_marker_length_[end]+lane_marker_x[end] > W? W-lane_marker_x[end]:lane_marker_length_[end]
		lane_marker_x[1] = max(0.,lane_marker_x[1])
		lane_marker_y = collect(pp.w_lane:pp.w_lane:(H-(pp.w_lane/2.)))

		for (i,x) in enumerate(lane_marker_x)
	    for y in lane_marker_y
	        draw_box(x,y+Y_ABS,lane_marker_length_[i],lane_marker_width,"#F2F2F2") #off-white
	    end
		end

	end
	#println("c")

	#draw self car
	#TODO
	#x_ctr = W/2. + frame
  x_ctr = s.env_cars[1].x + frame
	y_ctr = pp.y_interval*s.env_cars[1].y - ROW_INTERVAL*floor(Integer,x_ctr/lane_length)
	x_ctr = mod(x_ctr,lane_length)
	color = "#31B404" #PLACEHOLDER, an awful lime green
	draw_sedan(pp,x_ctr,y_ctr,color)
	####TODO: move this to a separate function, but it's here for now because I'm lazy
	hw = 0.5*pp.w_car
	hl = 1.*hw
	w = 0.75*hw
	dx = a.acc*2.5*hl
	dy = a.lane_change*pp.y_interval
	dy = dy != 0. ? dy - sign(dy)*hl: dy
	arrow(x_ctr,y_ctr,dx,dy,width=w,head_width=hw,head_length=hl,fc="#DF7401", ec="#0404B4",alpha=0.75)
	####END TODO
  v_nom = s.env_cars[1].vel
	annotate("$(round(v_nom,2))",xy=(x_ctr,y_ctr))

	#draw environment cars
	for car in s.env_cars[2:end]
		draw_sedan(pp,car,v_nom,frame,ROW_INTERVAL)
	end
	#println("e")

	#if is_crash, add crash graphic
    if is_crash(mdp,s,sp,debug)
  		draw_bang(x_ctr,y_ctr)
    end
	axis("equal")
	xlim(-pp.l_car,W)
	return gcf()
end
