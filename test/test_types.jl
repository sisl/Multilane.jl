##MDP Types
#Convenience abstract types to remove dependence on external packages for hte purposes of testing
function test_behavior_model_creation()
	println("\t\tTesting IDMMOBILBehavior creation")
	ps = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[27.,31.,35.],[4.]))]
end

function test_behavior_model_equality()
	ps = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[27.,31.,35.],[4.]))]
	test_equality("IDMMOBILBehavior",ps)
end

function test_behavior_model_hashing()
	ps = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[27.,31.,35.],[4.]))]
	test_hashing("IDMMOBILBehavior",ps)
end

function test_carstate_creation()
	println("\t\tTesting CarState creation")
	#nothing to do here--no constructors to test
end

function test_carstate_equality()
	#println("\t\tTesting CarState equality")
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[27.,31.,35.],[4.]))]
	ps = CarState[]
	push!(ps,CarState(1.,1,3.,0,bs[1],0))
	push!(ps,CarState(1.,1,3.,0,bs[2],0))
	push!(ps,CarState(1.,1,5.,0,bs[1],0))
	push!(ps,CarState(1.,1,3.,1,bs[1],0))
	push!(ps,CarState(1.,2,3.,0,bs[1],0))
	push!(ps,CarState(2.,1,3.,0,bs[1],0))

	test_equality("CarState",ps)

end

function test_carstate_hashing()
	#println("\t\tTesting CarState hashing")
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[27.,31.,35.],[4.]))]
	ps = CarState[]
	push!(ps,CarState(1.,1,3.,0,bs[1],0))
	push!(ps,CarState(1.,1,3.,0,bs[2],0))
	push!(ps,CarState(1.,1,5.,0,bs[1],0))
	push!(ps,CarState(1.,1,3.,1,bs[1],0))
	push!(ps,CarState(1.,2,3.,0,bs[1],0))
	push!(ps,CarState(2.,1,3.,0,bs[1],0))

	test_hashing("CarState",ps)
end

function test_MLState_creation()
	println("\t\tTesting MLState creation")
	#nothing to do here--no constructors to test
end

function test_MLState_equality()
	#println("\t\tTesting MLState equality")
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[27.,31.,35.],[4.]))]
	cs = CarState[]
	push!(cs,CarState(1.,1,3.,0,bs[1],0))
	push!(cs,CarState(1.,1,3.,0,bs[2],0))
	push!(cs,CarState(1.,1,5.,0,bs[1],0))
	push!(cs,CarState(1.,1,3.,1,bs[1],0))
	push!(cs,CarState(1.,2,3.,0,bs[1],0))
	push!(cs,CarState(2.,1,3.,0,bs[1],0))

	ps = MLState[]
	push!(ps,MLState(false,0.0,0.0,2,5.,cs[1:1]))
	push!(ps,MLState(false,0.0,0.0,3,5.,cs[1:1]))
	push!(ps,MLState(false,0.0,0.0,2,4.,cs[1:1]))
	push!(ps,MLState(false,0.0,0.0,2,5.,cs[1:2]))
	push!(ps,MLState(true,0.0,0.0,2,5.,cs[1:2]))

	test_equality("MLState",ps)
end

function test_MLState_hashing()
	#println("\t\tTesting MLState hashing")
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[27.,31.,35.],[4.]))]
	cs = CarState[]
	push!(cs,CarState(1.,1,3.,0,bs[1],0))
	push!(cs,CarState(1.,1,3.,0,bs[2],0))
	push!(cs,CarState(1.,1,5.,0,bs[1],0))
	push!(cs,CarState(1.,1,3.,1,bs[1],0))
	push!(cs,CarState(1.,2,3.,0,bs[1],0))
	push!(cs,CarState(2.,1,3.,0,bs[1],0))

	ps = MLState[]
	push!(ps,MLState(false,0.0,0.0,2,5.,cs[1:1]))
	push!(ps,MLState(false,0.0,0.0,3,5.,cs[1:1]))
	push!(ps,MLState(false,0.0,0.0,2,4.,cs[1:1]))
	push!(ps,MLState(false,0.0,0.0,2,5.,cs[1:2]))
	push!(ps,MLState(true,0.0,0.0,2,5.,cs[1:2]))

	test_hashing("MLState",ps)

    # all crashed states should be the same
    assert(MLState(true,0.0,0.0,3,5.,cs[1:2]), MLState(true,0.0,0.0,3,4.,cs[1:2]))
end

function test_MLAction_creation()
	println("\t\tTesting MLAction creation")
	#nothing to do here--no constructors to test
	#as = [MLAction(x[1],x[2]) for x in product([-1;0;1],[-1;0;1])]
end

function test_MLAction_equality()
	as = [MLAction(x[1],x[2]) for x in product([-1;0;1],[-1;0;1])]
	test_equality("MLAction",as)
end

function test_MLAction_hashing()
	as = [MLAction(x[1],x[2]) for x in product([-1;0;1],[-1;0;1])]
	test_hashing("MLAction",as)
end

function test_MLObs_creation()
	println("\t\tTesting MLObs creation")
	#nothing to do here--no constructors to test
end

function test_MLMDP_creation()
	println("\t\tTesting OriginalMDP type creation")
	p = OriginalMDP()
end

function test_n_state()
	println("\t\tTesting n_states")
	pp = PhysicalParam(2,nb_vel_bins=8,lane_length=2.5) #2.5=>10
	p = OriginalMDP(nb_cars=1,phys_param=pp)
	p.nb_col = 4
	assert(n_states(p) == (4*8)*(4*10*8*3*9+1))
	#idk if its worht it to do more
end

function test_repr()
    println("\t\tTesting the repr() function")
    bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[27.,31.,35.],[4.]))]
	c = CarState(1.,1,3.,0,bs[1],0)
    c2 = eval(parse(repr(c)))
    @test isa(c2, CarState)
end

#don't think you need tests for n_actions, the *Space types, and the associated domain() and length() functions...

function test_mdp_types()
	println("\tTesting MDP Type Units...")
	test_behavior_model_creation()
	test_behavior_model_equality()
	test_behavior_model_hashing()
	test_carstate_creation()
	test_carstate_equality()
	test_carstate_hashing()
	test_MLState_creation()
	test_MLState_equality()
	test_MLState_hashing()
	test_MLAction_creation()
	test_MLAction_equality()
	test_MLAction_hashing()
    test_repr()
end
