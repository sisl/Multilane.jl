abstract BehaviorModel
abstract AbstractMLRewardModel
abstract AbstractMLDynamicsModel

type MLMDP{S, A, DModel<:AbstractMLDynamicsModel, RModel<:AbstractMLRewardModel}  <: MDP{S, A}
    dmodel::DModel
    rmodel::RModel
    discount::Float64
end

type MLPOMDP{S, A, O, DModel<:AbstractMLDynamicsModel, RModel<:AbstractMLRewardModel}  <: POMDP{S, A, O}
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
    BEHAVIORS = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(Iterators.product(["cautious","normal","aggressive"],[phys_param.v_slow+0.5;phys_param.v_med;phys_param.v_fast],[phys_param.l_car]))]
    return IDMMOBILModel(nb_cars, phys_param, BEHAVIORS, length(BEHAVIORS), encounter_prob, accels)
end

# TODO for performance, parameterize this by BehaviorModel
immutable CarState
    x::Float64
    y::Float64
	vel::Float64 #v_x
	lane_change::Float64 # ydot # in units of LANES PER SECOND
	behavior::BehaviorModel
    id::Int # car id to track from state to state - ego is ALWAYS 1
end

function ==(a::CarState,b::CarState)
    return a.x==b.x && a.y==b.y && a.vel==b.vel && a.lane_change == b.lane_change && a.id == b.id && a.behavior==b.behavior
end
Base.hash(a::CarState, h::UInt64=zero(UInt64)) = hash(a.vel, hash(a.x, hash(a.y, hash(a.lane_change, hash(a.behavior, hash(a.id, h))))))
"Return a representation that will produce a valid object if executed"
Base.repr(c::CarState) = "CarState($(c.x),$(c.y),$(c.vel),$(c.lane_change),$(c.behavior),$(c.id))"

type MLState
    crashed::Bool # A crash occurs at the state transition. All crashed states are considered equal
    x::Float64 # total distance traveled by the ego
    t::Float64 # total time of the simulation
	cars::Array{CarState,1} #NOTE ego car is first car
end
# more constructors at bottom

function ==(a::MLState, b::MLState)
    if a.crashed && b.crashed
        return true
    elseif a.crashed || b.crashed # only one has crashed
        return false
    end
    return a.x == b.x && a.t == b.t && a.cars == b.cars #&& (a.agent_pos==b.agent_pos) && (a.agent_vel==b.agent_vel)
end
function Base.hash(a::MLState, h::UInt64=zero(UInt64))
    if a.crashed
        return hash(a.crashed, h)
    end
    return hash(a.x, hash(a.t, hash(a.cars,h)))#hash(a.agent_vel,hash(a.agent_pos,hash(a.cars,h)))
end

#= # don't think we need a special repr anymore since the behaviors are not nullable
function Base.repr(s::MLState)
    rstring = "MLState($(s.crashed), CarState["
    for (i,c) in enumerate(s.cars)
        rstring = string(rstring, "$(repr(c))")
        if i < length(s.cars)
            rstring = string(rstring, ",")
        end
    end
    return string(rstring, "])")
end
=#

immutable MLAction
    acc::Float64
    lane_change::Float64 # ydot
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

immutable CarPhysicalState
    x::Float64
    y::Float64
    vel::Float64
    lane_change::Float64
    id::Int
end
typealias CarStateObs CarPhysicalState

==(a::CarPhysicalState, b::CarPhysicalState) = (a.x == b.x) && (a.y == b.y) && (a.vel == b.vel) && (a.lane_change == b.lane_change) && (a.id == b.id)
Base.hash(a::CarPhysicalState, h::UInt64=zero(UInt64)) = hash(a.x, hash(a.y, hash(a.vel, (hash(a.lane_change, hash(a.id,h))))))
CarPhysicalState(cs::CarState) = CarPhysicalState(cs.x, cs.y, cs.vel, cs.lane_change, cs.id)
function CarState(cps::CarPhysicalState, behavior::BehaviorModel)
    return CarState(cps.x, cps.y, cps.vel, cps.lane_change, behavior, cps.id)
end

immutable MLPhysicalState
    crashed::Bool
    x::Float64
    t::Float64
    cars::Array{CarPhysicalState,1}
end
typealias MLObs MLPhysicalState

MLPhysicalState(s::MLState) = MLPhysicalState(s.crashed, s.x, s.t, CarPhysicalState[CarPhysicalState(cs) for cs in s.cars])
function ==(a::MLPhysicalState, b::MLPhysicalState)
    if a.crashed && b.crashed
        return true
    elseif a.crashed || b.crashed # only one has crashed
        return false
    end
    return a.x == b.x && a.t == b.t && a.cars == b.cars #&& (a.agent_pos==b.agent_pos) && (a.agent_vel==b.agent_vel)
end
function Base.hash(a::MLPhysicalState, h::UInt64=zero(UInt64))
    if a.crashed
        return hash(a.crashed, h)
    end
    return hash(a.x, hash(a.t, hash(a.cars,h))) #hash(a.agent_vel,hash(a.agent_pos,hash(a.cars,h)))
end

MLState(ps::MLPhysicalState, cars::Vector{CarState}) = MLState(ps.crashed, ps.x, ps.t, cars)
MLState(s::MLState, cars::Vector{CarState}) = MLState(s.crashed, s.x, s.t, cars)

# below are for tests only
function MLState(pos::Real, vel::Real, cars::Array{CarState,1}, x::Real=50.)
    #x = mdp.phys_param.lane_length/2.
    insert!(cars,1,CarState(x, pos, vel, 0, NORMAL, 0))
    return MLState(false, 0.0, 0.0, cars)
end
function MLState(crashed::Bool,pos::Real,vel::Real,cars::Array{CarState,1},x::Real=50.)
    insert!(cars,1,CarState(x, pos, vel, 0, NORMAL, 0))
    return MLState(crashed, 0.0, 0.0, cars)
end
function MLState(crashed::Bool,x::Float64,t::Float64,pos::Real,vel::Real,cars::Array{CarState,1},x_ego::Real=50.)
    insert!(cars,1,CarState(x_ego, pos, vel, 0, NORMAL, 0))
    return MLState(crashed, x, t, cars)
end
