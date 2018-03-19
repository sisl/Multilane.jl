abstract type BehaviorModel end
abstract type AbstractMLRewardModel end
abstract type AbstractMLDynamicsModel end

mutable struct MLMDP{S, A, DModel<:AbstractMLDynamicsModel, RModel<:AbstractMLRewardModel}  <: MDP{S, A}
    dmodel::DModel
    rmodel::RModel
    discount::Float64
end

mutable struct MLPOMDP{S, A, O, DModel<:AbstractMLDynamicsModel, RModel<:AbstractMLRewardModel}  <: POMDP{S, A, O}
  dmodel::DModel
  rmodel::RModel
  discount::Float64
end

mutable struct OriginalRewardModel <: AbstractMLRewardModel
	r_crash::Float64
	accel_cost::Float64
	decel_cost::Float64
	invalid_cost::Float64
	lineride_cost::Float64
	lanechange_cost::Float64
end

mutable struct IDMMOBILModel <: AbstractMLDynamicsModel
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
struct CarState
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

mutable struct MLState
    x::Float64 # total distance traveled by the ego
    t::Float64 # total time of the simulation
	cars::Array{CarState,1} #NOTE ego car is first car
    terminal::Nullable{Any} # SHOULD BE Nullable{Symbol} if not null, this is a terminal state, see below
end
# more constructors at bottom

#=
Terminal states: Each terminal state is not considered different, ther terminal states are
    :crash
    :lane
    :brake
    :distance
=#

function ==(a::MLState, b::MLState)
    if isnull(a.terminal) && isnull(b.terminal) # neither terminal
        return a.x == b.x && a.t == b.t && a.cars == b.cars
    elseif !isnull(a.terminal) && !isnull(b.terminal) # both terminal
        return get(a.terminal) == get(b.terminal)
    else # one is terminal
        return false
    end
end
function Base.hash(a::MLState, h::UInt64=zero(UInt64))
    if isnull(a.terminal)
        return hash(a.x, hash(a.t, hash(a.cars,h)))
    else
        return hash(get(a.terminal), h)
    end
end

struct MLAction
    acc::Float64
    lane_change::Float64 # ydot
end
MLAction() = MLAction(0,0)
==(a::MLAction,b::MLAction) = (a.acc==b.acc) && (a.lane_change==b.lane_change)
Base.hash(a::MLAction,h::UInt64=zero(UInt64)) = hash(a.acc,hash(a.lane_change,h))
function MLAction(x::Array{Float64,1})
	assert(length(x)==2)
	lane_change = abs(x[2]) <= 0.3 ? 0 : sign(x[2])
	return MLAction(x[1],lane_change)
end
vec(a::MLAction) = Float64[a.acc;a.lane_change]

const OriginalMDP = MLMDP{MLState, MLAction, IDMMOBILModel, OriginalRewardModel}

mutable struct ActionSpace
	actions::Vector{MLAction}
end

struct CarPhysicalState
    x::Float64
    y::Float64
    vel::Float64
    lane_change::Float64
    id::Int
end
const CarStateObs = CarPhysicalState

==(a::CarPhysicalState, b::CarPhysicalState) = (a.x == b.x) && (a.y == b.y) && (a.vel == b.vel) && (a.lane_change == b.lane_change) && (a.id == b.id)
Base.hash(a::CarPhysicalState, h::UInt64=zero(UInt64)) = hash(a.x, hash(a.y, hash(a.vel, (hash(a.lane_change, hash(a.id,h))))))
CarPhysicalState(cs::CarState) = CarPhysicalState(cs.x, cs.y, cs.vel, cs.lane_change, cs.id)
function CarState(cps::CarPhysicalState, behavior::BehaviorModel)
    return CarState(cps.x, cps.y, cps.vel, cps.lane_change, behavior, cps.id)
end

struct MLPhysicalState
    x::Float64
    t::Float64
    cars::Array{CarPhysicalState,1}
    terminal::Nullable{Any} # Should be Nullable{Symbol}
end
const MLObs = MLPhysicalState

MLPhysicalState(s::MLState) = MLPhysicalState(s.x, s.t, CarPhysicalState[CarPhysicalState(cs) for cs in s.cars], s.terminal)

function ==(a::MLPhysicalState, b::MLPhysicalState)
    if isnull(a.terminal) && isnull(b.terminal) # neither terminal
        return a.x == b.x && a.t == b.t && a.cars == b.cars
    elseif !isnull(a.terminal) && !isnull(b.terminal) # both terminal
        return get(a.terminal) == get(b.terminal)
    else # one is terminal
        return false
    end
end
function Base.hash(a::MLPhysicalState, h::UInt64=zero(UInt64))
    if isnull(a.terminal)
        return hash(a.x, hash(a.t, hash(a.cars,h)))
    else
        return hash(get(a.terminal), h)
    end
end

MLState(ps::MLPhysicalState, cars::Vector{CarState}) = MLState(ps.x, ps.t, cars, ps.terminal)
MLState(s::MLState, cars::Vector{CarState}) = MLState(s.x, s.t, cars, s.terminal)
MLState(x::Float64, t::Float64, cars::Vector{CarState}) = MLState(x, t, cars, nothing)
