function actions_mdp_fixture()
    dmodel = NoCrashIDMMOBILModel(3, PhysicalParam(3, lane_length=100.0))
    rmodel = NoCrashRewardModel()
    return NoCrashMDP{typeof(rmodel)}(dmodel, rmodel, 1.0, true)
end

function test_iteration()
    println("\t\tTesting Action iteration.")
    mdp = actions_mdp_fixture()    
    s = MLState(0.0,0.0, CarState[CarState(50.0, 2.0, 30.0, 0.0, Multilane.NORMAL, 0)])
    as = actions(mdp)
    as = actions(mdp, s, as)
    @test length(collect(as)) == length(as.acceptable)+1
end

function test_off_road()
    println("\t\tTesting No Offroad Actions...")
    mdp = actions_mdp_fixture()
    as = actions(mdp)
    right = MLState(0.0,0.0, CarState[CarState(50.0, 1.0, 30.0, 0.0, Multilane.NORMAL, 0)])
    ar = collect(actions(mdp, right, as))
    @test length(ar) > 1
    @test all(a -> a.lane_change >= 0., ar)
    left = MLState(0.0,0.0, CarState[CarState(50.0, 3.0, 30.0, 0.0, Multilane.NORMAL, 0)])
    al = collect(actions(mdp, left, as))
    @test length(al) > 1
    @test all(a -> a.lane_change <= 0., al)
end

function test_all_safe()
    println("\t\tTesting case where all actions are safe")
    mdp = actions_mdp_fixture()
    as = actions(mdp)
    allsafe = MLState(0.0,0.0, CarState[CarState(50.0, 2.0, 30.0, 0.0, Multilane.NORMAL, 0)])
    @test length(collect(actions(mdp, allsafe, as))) == 10
end

function failure_4()
    println("\t\tTesting special failure case changing into new lane")
    mdp = Multilane.MLMDP{Multilane.MLState,Multilane.MLAction,Multilane.NoCrashIDMMOBILModel,Multilane.NoCrashRewardModel}(Multilane.NoCrashIDMMOBILModel(10,Multilane.PhysicalParam(0.75,2.0,4.0,31.0,4.0,35.0,27.0,31.0,35.0,27.0,20.0,4,100.0),Multilane.BehaviorModel[Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,27.5,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),1),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,27.5,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),2),Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,27.5,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),3),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,31.0,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),4),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,31.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),5),Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,31.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),6),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,35.0,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),7),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,35.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),8),Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,35.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),9)],StatsBase.WeightVec{Float64,Array{Float64,1}}([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],9.0),1.0,1.3333333333333333,0.5,20.0,0.5,[1.0,1.0,1.0,1.0],2.0,false),Multilane.NoCrashRewardModel(100.0,10.0,5.0,4),1.0)

    s = MLState(false,0.0,0.0, CarState[CarState(50.0,3.0,31.22970781020354,0.0,Multilane.NORMAL,1),CarState(99.24797069990433,4.0,27.167643727404155,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,27.5,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),3),2)])

    @test Multilane.is_safe(mdp, s, MLAction(0.0, 1.0))
end

function failure_5()
    println("\t\tTesting special failure case allowing dangerous lane change when abs(lane_change) > 1")
    mdp = Multilane.MLMDP{Multilane.MLState,Multilane.MLAction,Multilane.NoCrashIDMMOBILModel,Multilane.NoCrashRewardModel}(Multilane.NoCrashIDMMOBILModel(10,Multilane.PhysicalParam(0.75,2.0,4.0,31.0,4.0,35.0,27.0,31.0,35.0,27.0,20.0,4,100.0),Multilane.BehaviorModel[Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,27.5,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),1),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,27.5,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),2),Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,27.5,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),3),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,31.0,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),4),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,31.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),5),Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,31.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),6),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,35.0,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),7),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,35.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),8),Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,35.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),9)],StatsBase.WeightVec{Float64,Array{Float64,1}}([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],9.0),1.0,2.6666666666666665,0.5,20.0,0.5,[1.0,1.0,1.0,1.0],2.0,false),Multilane.NoCrashRewardModel(100.0,10.0,8.0,1),1.0)
    state = MLState(false,0.0,0.0, CarState[CarState(50.0,2.0,27.0,0.0,Multilane.NORMAL,1),CarState(52.934068808019056,1.0,32.940581978786135,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,35.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),8),3),CarState(7.329881331042117,1.5,27.127595226858716,-0.6666666666666666,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,35.0,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),7),8),CarState(1.3070886107784006,3.0,27.306814286745862,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,31.0,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),4),10)])
    as = actions(mdp, state, actions(mdp))
    @test all([a.lane_change >= 0.0 for a in as])
end

function failure_6()
    println("\t\tTesting special failure case changing into a car slightly behind.")
    mdp = Multilane.MLMDP{Multilane.MLState,Multilane.MLAction,Multilane.NoCrashIDMMOBILModel,Multilane.NoCrashRewardModel}(Multilane.NoCrashIDMMOBILModel(10,Multilane.PhysicalParam(0.75,2.0,4.0,31.0,4.0,35.0,27.0,31.0,35.0,27.0,8.0,4,100.0),Multilane.CopulaIDMMOBIL(Multilane.IDMParam(0.7999999999999999,1.0,1.0,27.669999999999998,0.0,4.0),Multilane.IDMParam(2.0,3.0,2.0,38.93,4.0,4.0),Multilane.MOBILParam(0.0,1.0,0.0),Multilane.MOBILParam(1.0,3.0,0.2),Multilane.GaussianCopula([1.0 0.25 0.25 0.25 0.25 0.25 0.25 0.25; 0.25 1.0 0.25 0.25 0.25 0.25 0.25 0.25; 0.25 0.25 1.0 0.25 0.25 0.25 0.25 0.25; 0.25 0.25 0.25 1.0 0.25 0.25 0.25 0.25; 0.25 0.25 0.25 0.25 1.0 0.25 0.25 0.25; 0.25 0.25 0.25 0.25 0.25 1.0 0.25 0.25; 0.25 0.25 0.25 0.25 0.25 0.25 1.0 0.25; 0.25 0.25 0.25 0.25 0.25 0.25 0.25 1.0],[1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.25 0.968246 0.0 0.0 0.0 0.0 0.0 0.0; 0.25 0.193649 0.948683 0.0 0.0 0.0 0.0 0.0; 0.25 0.193649 0.158114 0.935414 0.0 0.0 0.0 0.0; 0.25 0.193649 0.158114 0.133631 0.92582 0.0 0.0 0.0; 0.25 0.193649 0.158114 0.133631 0.115728 0.918559 0.0 0.0; 0.25 0.193649 0.158114 0.133631 0.115728 0.102062 0.912871 0.0; 0.25 0.193649 0.158114 0.133631 0.115728 0.102062 0.0912871 0.908295]),377605),1.0,0.6666666666666666,1.0,35.0,0.25,false,4.0,Inf),Multilane.NoCrashRewardModel(100.0,10.0,8.0,1),1.0, true)
    state = Multilane.MLState(0.0,0.0,Multilane.CarState[Multilane.CarState(50.0,2.0,30.9774,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.86615,2.77692,1.11154,37.674,0.446166,4.0),Multilane.MOBILParam(0.111542,2.77692,0.0223083),371338),1),Multilane.CarState(45.9956,3.0,30.9545,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.27863,1.79771,1.60114,32.1611,2.40458,4.0),Multilane.MOBILParam(0.601144,1.79771,0.120229),371341),16)],Nullable{Any}())
    as = actions(mdp, state, actions(mdp))
    # if !all([a.lane_change <= 0.0 for a in as])
    #     Gallium.@enter actions(mdp, state)
    # end
    @test all([a.lane_change <= 0.0 for a in as])
end

function test_actions()
    println("\tTesting Action Space...")
    test_iteration()
    test_off_road()
    test_all_safe()
    # failure_4() # removed because of behavior change
    # failure_5() # removed because of behavior change
    failure_6()
end
