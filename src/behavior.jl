type IDMMOBILBehavior <: BehaviorModel
	p_idm::IDMParam
	p_mobil::MOBILParam
	rationality::Float64
	idx::Int
end
==(a::IDMMOBILBehavior,b::IDMMOBILBehavior) = (a.p_idm==b.p_idm) && (a.p_mobil==b.p_mobil) &&(a.rationality == b.rationality)
Base.hash(a::IDMMOBILBehavior,h::UInt64=zero(UInt64)) = hash(a.p_idm,hash(a.p_mobil,hash(a.rationality,h)))

function IDMMOBILBehavior(s::AbstractString,v0::Float64,s0::Float64,idx::Int)
	typedict = Dict{AbstractString,Float64}("cautious"=>1.,"normal"=>1.,"aggressive"=>1.) #rationality
	return IDMMOBILBehavior(IDMParam(s,v0,s0),MOBILParam(s),typedict[s],idx)
end

