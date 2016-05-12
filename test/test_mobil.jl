function test_mobil_creation()
	println("\t\tTesting MOBIL Creation")
	p1 = MOBILParam()
	p2 = MOBILParam(p=0.5)
	p3 = MOBILParam(b_safe=5.,a_thr=0.3)
	ps = [MOBILParam(s) for s in ["cautious";"normal";"aggressive"]]
end

function test_mobil_equality()
	#println("\t\tTesting MOBIL Equality")
	p1 = MOBILParam()
	p2 = MOBILParam(p=0.5)
	p3 = MOBILParam(b_safe=5.,a_thr=0.3)
	ps = [MOBILParam(s) for s in ["cautious";"normal";"aggressive"]]
	test_equality("MOBIL",ps)
	assert(p1 == p1)
	assert(p1 != p2)
	assert(p2 != p1)
	assert(p1 == MOBILParam(0.25,4.,0.2))
	assert(MOBILParam(0.25,4.,0.2) == p1)
end

function test_mobil_hashing()
	println("\t\tTesting MOBIL Hashing")
	p1 = MOBILParam()
	p2 = MOBILParam(p=0.5)
	p3 = MOBILParam(b_safe=5.,a_thr=0.3)
	ps = [MOBILParam(s) for s in ["cautious";"normal";"aggressive"]]
	test_hashing("MOBIL",ps)
end

function test_get_neighborhood()
	println("\t\tTesting get_neighborhood")
	##TODO: assertions
	nb_lanes = 3
	pp = PhysicalParam(nb_lanes,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#CASE: just agent car
	cs = CarState[CarState(6.,2,31.,0,bs[1],1)]
	dp1 = 6.-pp.l_car
	dp2 = 6.-pp.l_car
	_a = MLAction(0,0)

	#just make sure it doesn't explode--this should be handled by get_mobil_lane_change
	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,1)
	assert(length(nbhd) == 6)
	for jdx in nbhd
		assert(jdx == 0)
	end
	#CASE: nobody to the left
	cs = CarState[CarState(12.,2,31.,0,bs[1],2),CarState(6.,2,31.,0,bs[1],3),CarState(0.,2,31.,0,bs[1],4),CarState(12.,1,31.,0,bs[1],5),CarState(0.,1,31.,0,bs[1],6)]

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,2)
	assert(length(nbhd) == 6)
	for jdx in nbhd[[3;6]]
		assert(jdx == 0)
	end
	assert(nbhd[1] , 4)
	assert(nbhd[2] , 1)
	assert(nbhd[4] , 5)
	assert(nbhd[5] , 3)
	#CASE: nobody to the right
	cs = CarState[CarState(12.,2,31.,0,bs[1],7),CarState(0.,2,31.,0,bs[1],8),CarState(6.,2,31.,0,bs[1],9),CarState(12.,3,31.,0,bs[1],10),CarState(0.,3,31.,0,bs[1],11)]

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,3)
	assert(length(nbhd) == 6)
	for jdx in nbhd[[1;4]]
		assert(jdx , 0)
	end
	assert(nbhd[3] , 4)
	assert(nbhd[2] , 1)
	assert(nbhd[6] , 5)
	assert(nbhd[5] , 2)
	#CASE: no one ahead
	cs = CarState[CarState(0.,3,31.,0,bs[1],12),CarState(0.,2,31.,0,bs[1],13),CarState(0.,1,31.,0,bs[1],14),CarState(6.,2,31.,0,bs[1],15)]

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,4)
	assert(length(nbhd) , 6)
	for jdx in nbhd[[1;2;3]]
		assert(jdx , 0)
	end
	assert(nbhd[4] , 3)
	assert(nbhd[5] , 2)
	assert(nbhd[6] , 1)
	#CASE: no one behind
	cs = CarState[CarState(12.,3,31.,0,bs[1],16),CarState(12.,2,31.,0,bs[1],17),CarState(12.,1,31.,0,bs[1],18),CarState(6.,2,31.,0,bs[1],19)]

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,4)
	assert(length(nbhd) , 6)
	for jdx in nbhd[[4;5;6]]
		assert(jdx , 0)
	end
	assert(nbhd[1] , 3)
	assert(nbhd[2] , 2)
	assert(nbhd[3] , 1)
	#CASE: full house
	cs = CarState[CarState(12.,2,31.,0,bs[1],20),CarState(0.,2,31.,0,bs[1],21),CarState(12.,1,31.,0,bs[1],22),CarState(0.,1,31.,0,bs[1],23),CarState(12.,3,35.,0,bs[1],24),CarState(0.,3,35.,0,bs[1],25),CarState(6.,2,31.,0,bs[1],26)]

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,7)
	assert(length(nbhd) , 6)
	for jdx in nbhd
		assert(jdx != 0)
	end
	assert(nbhd[1] , 3)
	assert(nbhd[2] , 1)
	assert(nbhd[3] , 5)
	assert(nbhd[4] , 4)
	assert(nbhd[5] , 2)
	assert(nbhd[6] , 6)
end
#TODO test_get_dv_ds(); FIX VVV

function test_is_lanechange_dangerous()
	println("\t\tTesting is_lanechange_dangerous")

	nb_lanes = 3
	_a = MLAction(0,0)
	pp = PhysicalParam(nb_lanes,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#CASE: no one, in the way
	cs = CarState[CarState(6.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),27)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,1)
	assert(is_lanechange_dangerous(pp,s,nbhd,1,0),false)
	assert(is_lanechange_dangerous(pp,s,nbhd,1,1),false)
	assert(is_lanechange_dangerous(pp,s,nbhd,1,-1),false)

	#CASE: person close by, same velocity
	cs = CarState[CarState(6.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),28),CarState(12.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),29)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,1)
	assert(is_lanechange_dangerous(pp,s,nbhd,1,1),false)

	#slightly behind
	cs = CarState[CarState(6.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),30),CarState(0.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),31)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,1)
	assert(is_lanechange_dangerous(pp,s,nbhd,1,1),false)
	#CASE: person a little behind, slower
	cs = CarState[CarState(6.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),32),CarState(1.,3,32.,0,IDMMOBILBehavior("aggressive",35.,4.,1),33)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,1)
	assert(is_lanechange_dangerous(pp,s,nbhd,1,1),false)
	#CASE: person a little ahead, faster
	cs = CarState[CarState(6.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),34),CarState(1.,3,32.,0,IDMMOBILBehavior("aggressive",35.,4.,1),35)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,2)
	assert(is_lanechange_dangerous(pp,s,nbhd,2,-1),false)
	#CASE: person a little behind, faster
	cs = CarState[CarState(6.,2,32.,0,IDMMOBILBehavior("aggressive",35.,4.,1),36),CarState(1.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),37)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,1)
	assert(is_lanechange_dangerous(pp,s,nbhd,1,1),true)
	#CASE: person a little ahead, slower
	cs = CarState[CarState(6.,2,32.,0,IDMMOBILBehavior("aggressive",35.,4.,1),38),CarState(1.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),39)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,2)
	assert(is_lanechange_dangerous(pp,s,nbhd,2,-1),true)
	#CASE: next to each other
	cs = CarState[CarState(6.,2,32.,0,IDMMOBILBehavior("aggressive",35.,4.,1),40),CarState(6.,3,33.,0,IDMMOBILBehavior("aggressive",35.,4.,1),41)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,2)
	assert(is_lanechange_dangerous(pp,s,nbhd,2,-1),true)
end

function test_get_rear_accel()
	println("\t\tTesting get_rear_accel")
	nb_lanes = 3
	_a = MLAction(0,0)
	pp = PhysicalParam(nb_lanes,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#No one in that position
	cs = CarState[CarState(6.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),42)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,1)
	a,a_ = get_rear_accel(pp,s,nbhd,1,0)
	assert(a,0.)
	assert(a_,0.)
	a,a_ = get_rear_accel(pp,s,nbhd,1,1)
	assert(a,0.)
	assert(a_,0.)
	a,a_ = get_rear_accel(pp,s,nbhd,1,-1)
	assert(a,0.)
	assert(a_,0.)
	#will go to free flow
	cs = CarState[CarState(6.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),42),
								CarState(0.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),43),
								CarState(0.,2,31.,0,IDMMOBILBehavior("aggressive",31.,4.,1),44),
								CarState(0.,1,27.,0,IDMMOBILBehavior("aggressive",27.,4.,1),45)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,1)
	a,a_ = get_rear_accel(pp,s,nbhd,1,0)
	assert(a_,a,>=)
	a,a_ = get_rear_accel(pp,s,nbhd,1,1)
	assert(a_,a,>=)
	a,a_ = get_rear_accel(pp,s,nbhd,1,-1)
	assert(a_,a,>=)
	#some synthetic case
end

function test_get_mobil_lane_change()
	println("\t\tTesting get_mobil_lane_change")
	nb_lanes = 2
	_a = MLAction(0,0)
	pp = PhysicalParam(nb_lanes,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#CASE: it's faster, but there's no space--is this even a real case?
	#CASE: it's faster and there is space
	cs = CarState[CarState(12.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),1),
								CarState(11.,1,27.,0,IDMMOBILBehavior("cautious",27.,4.,1),2),
								CarState(6.,1,31.,0,IDMMOBILBehavior("aggressive",35.,4.,1),3)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,3)
	assert(get_mobil_lane_change(pp,s,nbhd,3),1)
	#CASE: it's slower and there is space
	cs = CarState[CarState(12.,2,31.,0,IDMMOBILBehavior("cautious",31.,4.,1),6),CarState(11.,1,35.,0,IDMMOBILBehavior("cautious",35.,4.,1),5),CarState(6.,1,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),4)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,3)
	assert(get_mobil_lane_change(pp,s,nbhd,3),0)
	#CASE: someone is going fast behind me and i'm slow
	cs = CarState[CarState(0.,1,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),7),CarState(6.,1,27.,0,IDMMOBILBehavior("cautious",27.,4.,1),8)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,2)
	assert(get_mobil_lane_change(pp,s,nbhd,2),1)
	###repeat for other side
	#CASE: it's faster and there is space
	cs = CarState[CarState(12.,1,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),9),CarState(11.,2,27.,0,IDMMOBILBehavior("cautious",27.,4.,1),10),CarState(6.,2,31.,0,IDMMOBILBehavior("aggressive",35.,4.,1),11)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,3)
	assert(get_mobil_lane_change(pp,s,nbhd,3),-1)
	#CASE: it's slower and there is space
	cs = CarState[CarState(12.,1,31.,0,IDMMOBILBehavior("cautious",31.,4.,1),0),CarState(11.,2,35.,0,IDMMOBILBehavior("cautious",35.,4.,1),0),CarState(6.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),0)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,3)
	assert(get_mobil_lane_change(pp,s,nbhd,3),0)
	#CASE: someone is going fast behind me and i'm slow
	cs = CarState[CarState(0.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),0),CarState(6.,2,27.,0,IDMMOBILBehavior("cautious",27.,4.,1),0)]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(pp,s,2)
	assert(get_mobil_lane_change(pp,s,nbhd,2),-1)
end

function failure_2()
    println("\t\tTesting specific failure: car runs into ego")
    nb_lanes = 4
    pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8
    _discount = 1.
    nb_cars=10
    rmodel = NoCrashRewardModel()
    dmodel = NoCrashIDMMOBILModel(nb_cars, pp)
    mdp = NoCrashMDP(dmodel, rmodel, _discount);
    rng = MersenneTwister(2)
    s = Multilane.MLState(false,[Multilane.CarState(50.0,1.0,27.0,0.0,Nullable{Multilane.BehaviorModel}(),0),Multilane.CarState(45.66875308699299,2.0,29.042803490064543,-0.6666666666666666,Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,35.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),1.0,9),0),Multilane.CarState(85.41074952313767,3.0,29.831781823255017,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,31.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),1.0,6),0),Multilane.CarState(13.78112065279241,3.0,27.0,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,35.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),1.0,8),0),Multilane.CarState(5.897872040018292,1.0,27.39955827605183,0.0,Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,35.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),1.0,8),0)])
    a = MLAction(0.0,0.0)
    sp, r = generate_sr(mdp, s, a, rng)
    @test !is_crash(mdp, s, sp)
end

function failure_3()
    println("\t\tTesting specific failure: cars lane change into one another")
    mdp = Multilane.MLMDP{Multilane.MLState,Multilane.MLAction,Multilane.NoCrashIDMMOBILModel,Multilane.NoCrashRewardModel}(Multilane.NoCrashIDMMOBILModel(10,Multilane.PhysicalParam(0.75,2.0,4.0,31.0,4.0,4.0,35.0,27.0,31.0,35.0,27.0,20.0,4,100.0),Multilane.BehaviorModel[Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,27.5,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),1.0,1),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,27.5,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),1.0,2),Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,27.5,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),1.0,3),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,31.0,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),1.0,4),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,31.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),1.0,5),Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,31.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),1.0,6),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,35.0,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),1.0,7),Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,35.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),1.0,8),Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,35.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),1.0,9)],StatsBase.WeightVec{Float64,Array{Float64,1}}([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],9.0),1.0,1.3333333333333333,0.5,20.0,0.5,[1.0,1.0,1.0,1.0],2.0),Multilane.NoCrashRewardModel(100.0,10.0,1.5,1),1.0)
    s = Multilane.MLState(false,[Multilane.CarState(50.0,2.5,30.75,0.0,Nullable{Multilane.BehaviorModel}(),1),Multilane.CarState(99.56731292388635,3.0,34.36785842369873,0.0,Nullable(Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,35.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),1.0,9)),4),Multilane.CarState(88.9857138446193,4.0,29.878676022074934,0.0,Nullable(Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,31.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),1.0,5)),12),Multilane.CarState(83.55003341854245,1.0,27.45737686175689,0.0,Nullable(Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,27.5,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),1.0,2)),13),Multilane.CarState(5.833404602969644,1.5,31.164522812071382,0.6666666666666666,Nullable(Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,35.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),1.0,9)),14),Multilane.CarState(6.364105400258811,3.0,30.995876598243825,0.0,Nullable(Multilane.IDMMOBILBehavior(Multilane.IDMParam(2.0,2.0,0.8,31.0,4.0,4.0),Multilane.MOBILParam(0.0,4.0,0.2),1.0,6)),15),Multilane.CarState(95.2814275010719,2.0,27.0,0.0,Nullable(Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,31.0,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),1.0,5)),17),Multilane.CarState(98.12572985780923,2.0,29.049513872117117,0.0,Nullable(Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.0,1.0,2.0,27.5,4.0,4.0),Multilane.MOBILParam(0.5,4.0,0.2),1.0,1)),18),Multilane.CarState(0.0,4.0,31.09003055824878,0.0,Nullable(Multilane.IDMMOBILBehavior(Multilane.IDMParam(1.5,1.5,1.4,27.5,4.0,4.0),Multilane.MOBILParam(0.25,4.0,0.2),1.0,2)),19)])
    sp, r = generate_sr(mdp, s, MLAction(0,0), MersenneTwister(5))
    @test !Multilane.occupation_overlap(sp.env_cars[4].y, sp.env_cars[5].y)
end

function test_mobil()
	println("\tTesting MOBIL Units...")
	test_mobil_creation()
	test_mobil_equality()
	test_mobil_hashing()
	test_get_neighborhood()
	test_is_lanechange_dangerous()
	test_get_rear_accel()
	test_get_mobil_lane_change()
    failure_2()
    failure_3()
end
