abstract BehaviorGenerator

type BehaviorSet <: BehaviorGenerator
    behaviors::Vector{BehaviorModel}
    probabilities::WeighVec
end
