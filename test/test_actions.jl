function actions_mdp_fixture()
    dmodel = NoCrashIDMMOBILModel(3, PhysicalParam(3, lane_length=100.0))
    rmodel = NoCrashRewardModel()
    return NoCrashMDP(dmodel, rmodel, 1.0)
end

function test_iteration()
    println("\t\tTesting Action iteration.")
    mdp = actions_mdp_fixture()    
    s = MLState(false, CarState[CarState(50.0, 2.0, 30.0, 0.0, Nullable{BehaviorModel}(), 0)])
    as = actions(mdp)
    as = actions(mdp, s, as)
    @test length(collect(as)) == length(as.acceptable)+1
end

function test_off_road()
    println("\t\tTesting No Offroad Actions...")
    mdp = actions_mdp_fixture()
    as = actions(mdp)
    right = MLState(false, CarState[CarState(50.0, 1.0, 30.0, 0.0, Nullable{BehaviorModel}(), 0)])
    ar = collect(actions(mdp, right, as))
    @test length(ar) > 1
    @test all(a -> a.lane_change >= 0., ar)
    left = MLState(false, CarState[CarState(50.0, 3.0, 30.0, 0.0, Nullable{BehaviorModel}(), 0)])
    al = collect(actions(mdp, left, as))
    @test length(al) > 1
    @test all(a -> a.lane_change <= 0., al)
end

function test_all_safe()
    println("\t\tTesting case where all actions are safe")
    mdp = actions_mdp_fixture()
    as = actions(mdp)
    allsafe = MLState(false, CarState[CarState(50.0, 2.0, 30.0, 0.0, Nullable{BehaviorModel}(), 0)])
    @test length(collect(actions(mdp, allsafe, as))) == 10
end

# a specific failure I encountered
function failure_1()
    println("\t\tTesting a specific failure case")
    nb_lanes = 4
    pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8
    _discount = 1.
    nb_cars=10
    rmodel = NoCrashRewardModel()
    dmodel = NoCrashIDMMOBILModel(nb_cars, pp)
    mdp = NoCrashMDP(dmodel, rmodel, _discount)
    s = Multilane.MLState(false,[Multilane.CarState(50.0,1.0,35.0,0.0,Nullable{Multilane.BehaviorModel}(),0),Multilane.CarState(51.56670014233143,2.0,29.929470802139356,0.6666666666666666,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,31.0,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),1.0,4),1),Multilane.CarState(2.5160045443044012,3.5,34.09262003519827,0.6666666666666666,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,35.0,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),1.0,7),2),Multilane.CarState(95.67296223569727,1.5,31.33623262852605,0.6666666666666666,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,27.5,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),1.0,2),3),Multilane.CarState(99.62425695892622,3.0,34.37301855713661,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,35.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),1.0,9),4)])
    as = actions(mdp)
    as = actions(mdp, s, as)
    for a in as
        @test !is_crash(mdp, s, a)
    end
end

function test_actions()
    println("\tTesting Action Space...")
    test_iteration()
    test_off_road()
    test_all_safe()
    failure_1()
end
