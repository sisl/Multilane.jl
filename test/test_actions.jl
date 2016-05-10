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

function test_actions()
    println("\tTesting Action Space...")
    test_iteration()
    test_off_road()
    test_all_safe()
end
