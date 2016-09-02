using GenerativeModels
using Multilane
using Base.Test

mdp = Multilane.MLMDP{Multilane.MLState,Multilane.MLAction,Multilane.NoCrashIDMMOBILModel,Multilane.NoCrashRewardModel}(Multilane.NoCrashIDMMOBILModel(10,Multilane.PhysicalParam(0.75,2.0,4.0,31.0,4.0,35.0,27.0,31.0,35.0,27.0,8.0,4,100.0),Multilane.DiscreteBehaviorSet(Multilane.BehaviorModel[Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.4,2.0,1.5,33.333333333333336,2.0,4.0),Multilane.MOBILParam(0.4,2.0,0.1),1),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,1.8,27.77777777777778,4.0,4.0),Multilane.MOBILParam(0.4,1.0,0.1),2),Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,3.0,1.0,38.888888888888886,1.0,4.0),Multilane.MOBILParam(0.4,3.0,0.1),3)],StatsBase.WeightVec{Float64,Array{Float64,1}}([1.0,1.0,1.0],3.0)),1.0,0.6666666666666666,0.5,35.0,0.5,[1.0,1.0,1.0,1.0],100.0,false),Multilane.NoCrashRewardModel(100.0,10.0,2.5,4),1.0)

s = Multilane.MLState(false,[Multilane.CarState(50.0,1.0,31.74906513736733,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,3.0,1.0,38.888888888888886,1.0,4.0),Multilane.MOBILParam(0.4,3.0,0.1),3),1),Multilane.CarState(18.420260117169548,4.0,26.1928392571919,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,1.8,27.77777777777778,4.0,4.0),Multilane.MOBILParam(0.4,1.0,0.1),2),37),Multilane.CarState(49.36579094083517,3.0,27.477973049033686,0.6666666666666666,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,1.8,27.77777777777778,4.0,4.0),Multilane.MOBILParam(0.4,1.0,0.1),2),38),Multilane.CarState(90.12579426283317,3.5,28.861948465907133,0.6666666666666666,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,1.8,27.77777777777778,4.0,4.0),Multilane.MOBILParam(0.4,1.0,0.1),2),41),Multilane.CarState(100.0,1.0,31.578986008057452,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.4,2.0,1.5,33.333333333333336,2.0,4.0),Multilane.MOBILParam(0.4,2.0,0.1),1),42)])

a = Multilane.MLAction(-0.37277643831998186,0.6666666666666666)

sp = generate_s(mdp, s, a, MersenneTwister(1))

@test !(sp.cars[1].y > 1 && sp.cars[3].y < 3)

#=

using Cairo
c = visualize(mdp, s, a, sp)
fname = string(tempname(), ".png")
write_to_png(c, fname)
@spawn run(`xdg-open $fname`)
c2 = visualize(mdp, sp)
fname = string(tempname(), ".png")
write_to_png(c2, fname)
@spawn run(`xdg-open $fname`)

=#
