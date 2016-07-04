#######################
##Physical Parameters##
#######################

type PhysicalParam
	dt::Float64
	w_car::Float64
	l_car::Float64
	v_nominal::Float64
	w_lane::Float64
	v_fast::Float64
	v_slow::Float64
	v_med::Float64
	v_max::Float64
	v_min::Float64
    brake_limit::Float64
    nb_lanes::Int
	lane_length::Float64
end

function PhysicalParam(nb_lanes::Int;dt::Float64=0.75,
						w_car::Float64=2.,
						l_car::Float64=4.,
						v_nominal::Float64=31.,
						w_lane::Float64=4.,
						v_fast::Float64=35.,
						v_slow::Float64=27.,
						v_med::Float64=31.,
						lane_length::Float64=12.,
						v_max::Float64=v_fast+0.,
						v_min::Float64=v_slow-0.,
                        brake_limit::Float64=8. # coefficient of friction of about 0.8
                        )

	assert(v_fast >= v_med)
	assert(v_med >= v_slow)
	assert(v_fast > v_slow)
    return PhysicalParam(dt, w_car, l_car, v_nominal, w_lane, v_fast, v_slow, v_med, v_max, v_min, brake_limit, nb_lanes, lane_length)
end

"""
Returns true if cars at y1 and y2 occupy the same lane
"""
function occupation_overlap(y1::Float64, y2::Float64)
    return abs(y1-y2) < 1.0 || ceil(y1) == floor(y2) || floor(y1) == ceil(y2)
end
