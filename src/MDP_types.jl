abstract BehaviorModel
abstract AbstractMLRewardModel
abstract AbstractMLDynamicsModel

type MLMDP{S, A, DModel<:AbstractMLDynamicsModel, RModel<:AbstractMLRewardModel}  <: MDP{S, A}
    dmodel::DModel
    rmodel::RModel
	discount::Float64
end

type OriginalRewardModel <: AbstractMLRewardModel
	r_crash::Float64
	accel_cost::Float64
	decel_cost::Float64
	invalid_cost::Float64
	lineride_cost::Float64
	lanechange_cost::Float64
end

type IDMMOBILModel <: AbstractMLDynamicsModel
	nb_cars::Int
    phys_param::PhysicalParam

	BEHAVIORS::Array{BehaviorModel,1}
	NB_PHENOTYPES::Int

	encounter_prob::Float64
	accels::Array{Int,1}
end

function IDMMOBILModel(nb_cars, phys_param; encounter_prob=0.5, accels=Int[-3,-2,-1,0,1])
    BEHAVIORS = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[phys_param.v_slow+0.5;phys_param.v_med;phys_param.v_fast],[phys_param.l_car]))]
    return IDMMOBILModel(nb_cars, phys_param, BEHAVIORS, length(BEHAVIORS), encounter_prob, accels)
end

# TODO for performance, parameterize this by BehaviorModel
immutable CarState
	pos::Tuple{Float64,Float64} #row, col/ (x,y)
	vel::Float64 #v_x
	lane_change::Float64 #-1,0, or +1, corresponding to to the right lane, no lane change, or to the left lane (also v_y/heading)
	behavior::Nullable{BehaviorModel}

	function CarState(pos::Tuple{Float64,Real},vel::Float64,lane_change::Real,behavior::Union{Nullable{BehaviorModel},BehaviorModel})
		assert(abs(lane_change) <= 1.) #XXX might not be appropriate with new def of lane_change
		return new(pos, vel, lane_change, behavior)
	end
end #carstate
==(a::CarState,b::CarState) = (a.pos==b.pos) && (a.vel==b.vel) &&(a.lane_change == b.lane_change)&&((isnull(a.behavior) && isnull(b.behavior)) || (get(a.behavior)==get(b.behavior)))
Base.hash(a::CarState,h::UInt64=zero(UInt64)) = hash(a.vel,hash(a.pos,hash(a.lane_change,hash(a.behavior,h))))

type MLState
  crashed::Bool # A crash occurs at the state transition. All crashed states are considered equal
	#TODO goal lane?
	env_cars::Array{CarState,1} #NOTE ego car is first car
end #MLState
#XXX convenience functions
function MLState(pos::Real,vel::Real,cars::Array{CarState,1},x::Real=50.)
  #x = mdp.phys_param.lane_length/2.
  insert!(cars,1,CarState((x,pos),vel,0,Nullable{BehaviorModel}()))
  return MLState(false,cars)
end
function MLState(crashed::Bool,pos::Real,vel::Real,cars::Array{CarState,1},x::Real=50.)
  insert!(cars,1,CarState((x,pos),vel,0,Nullable{BehaviorModel}()))
  return MLState(crashed,cars)
end

function ==(a::MLState, b::MLState)
    if a.crashed && b.crashed
        return true
    elseif a.crashed || b.crashed # only one has crashed
        return false
    end
    return (a.env_cars == b.env_cars) #&& (a.agent_pos==b.agent_pos) && (a.agent_vel==b.agent_vel)
end
function Base.hash(a::MLState, h::UInt64=zero(UInt64))
    if a.crashed
        return hash(a.crashed, h)
    end
    return hash(a.env_cars,h)#hash(a.agent_vel,hash(a.agent_pos,hash(a.env_cars,h)))
end

immutable MLAction
	acc::Float64
  lane_change::Float64 #-1,0, or +1, corresponding to to the right lane, no lane change, or to the left lane
end
MLAction() = MLAction(0,0)
==(a::MLAction,b::MLAction) = (a.acc==b.acc) && (a.lane_change==b.lane_change)
Base.hash(a::MLAction,h::UInt64=zero(UInt64)) = hash(a.acc,hash(a.lane_change,h))
function MLAction(x::Array{Float64,1})
	assert(length(x)==2)
	lane_change = abs(x[2]) <= 0.3? 0: sign(x[2])
	return MLAction(x[1],lane_change)
end
vec(a::MLAction) = Float64[a.acc;a.lane_change]

typealias OriginalMDP MLMDP{MLState, MLAction, IDMMOBILModel, OriginalRewardModel}

type ActionSpace <: AbstractSpace
	actions::Vector{MLAction}
end
