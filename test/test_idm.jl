##IDM
function test_idm_creation()
	println("\t\tTesting IDM Creation")
	#do it directly using the build functions
	p1 = Multilane.build_cautious_idm(30.,2.)
	p2 = Multilane.build_normal_idm(30.,2.)
	p3 = Multilane.build_aggressive_idm(30.,2.)
	#use the new constructor
	ps = [IDMParam(s,35.,3.) for s in ["AGGRESSIVE","CautiOUS","normal"]]
end

function test_idm_equality()
	#do it directly using the build functions
	p1 = Multilane.build_cautious_idm(30.,2.)
	p2 = Multilane.build_normal_idm(30.,2.)
	p3 = Multilane.build_aggressive_idm(30.,2.)
	#use the new constructor
	ps = [IDMParam(s,35.,3.) for s in ["AGGRESSIVE","CautiOUS","normal"]]
	#test equality
	test_equality("IDM",ps)
	assert(p1 == IDMParam(1.,1.,2.,30.,2.))
	assert(p2 == IDMParam(1.5,1.5,1.4,30.,2.))
	assert(p3 == IDMParam(2.,2.,0.8,30.,2.))
	assert(IDMParam(1.,1.,2.,30.,2.) == p1)
	assert(p2 == p2)
	assert(ps[3] == ps[3])
	assert(p1 != ps[1])
	assert(p2 != ps[2])
	assert(p3 != ps[3])
	assert(p1 != p2)
	assert(p2 != p1)
end

function test_idm_hashing()
	#println("\t\tTesting IDM Hashing")
	#do it directly using the build functions
	p1 = Multilane.build_cautious_idm(30.,2.)
	p2 = Multilane.build_normal_idm(30.,2.)
	p3 = Multilane.build_aggressive_idm(30.,2.)
	#use the new constructor
	ps = [IDMParam(s,35.,3.) for s in ["AGGRESSIVE","CautiOUS","normal"]]
	#test hashing as applied to dictionaries
	test_hashing("IDM",ps)
end

#TODO FIX VVV

function test_get_dv_ds()
	println("\t\tTesting get_dv_ds()")
	nb_lanes = 3
	pp = PhysicalParam(nb_lanes,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#CASE: just agent car
	cs = CarState[CarState(6.,2,31.,0,bs[1],0)]
	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,false,0.0,0.0,cs)
	nbhd = get_neighborhood(pp,s,1)
	for nbr = 1:6
		dv, ds = get_dv_ds(pp,s,nbhd,1,nbr)
		assert(dv,0.)
		assert(ds,1000.)
	end
	#full house
	cs = CarState[CarState(0.,1,27.,0,bs[1],1),
								CarState(0.,2,30.,0,bs[1],2),
								CarState(0.,3,35.,0,bs[1],3),
								CarState(12.,1,27.,0,bs[1],4),
								CarState(12.,2,30.,0,bs[1],5),
								CarState(12.,3,35.,0,bs[1],6),
								CarState(6.,2,31.,0,bs[1],7)]

	_ds = 2.

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,false,0.0,0.0,cs)
	nbhd = get_neighborhood(pp,s,7)
	dv, ds = get_dv_ds(pp,s,nbhd,7,1)
	assert(dv,4.)
	assert(ds,_ds)
	dv, ds = get_dv_ds(pp,s,nbhd,7,2)
	assert(dv,1.)
	assert(ds,_ds)
	dv, ds = get_dv_ds(pp,s,nbhd,7,3)
	assert(dv,-4.)
	assert(ds,_ds)
	dv, ds = get_dv_ds(pp,s,nbhd,7,4)
	assert(dv,-4.)
	assert(ds,_ds)
	dv, ds = get_dv_ds(pp,s,nbhd,7,5)
	assert(dv,-1.)
	assert(ds,_ds)
	dv, ds = get_dv_ds(pp,s,nbhd,7,6)
	assert(dv,4.)
	assert(ds,_ds)

end

function test_idm_dv()
	println("\t\tTesting IDM Calculations")
	# NOTE this just prints it out for general sanity checking
	#TODO: calculate by hand...? or use existing examples if can be found
	nb_lanes = 2
	_a = MLAction(0.,0.)
	pp = PhysicalParam(nb_lanes,lane_length=48.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#CASE: it's faster, but there's no space--is this even a real case?
	#CASE: it's faster and there is space
	cs = CarState[CarState(1.,2,27.,0,IDMMOBILBehavior("aggressive",35.,4.,1),5),CarState(36.,2,27.,0,IDMMOBILBehavior("cautious",27.,4.,1),1)]
	s = MLState(false,false,0.0,0.0,cs)
	dmodel = IDMMOBILModel(length(cs),pp)
	nbhd = get_neighborhood(pp,s,1)
	v = cs[1].vel
	dv, ds = get_dv_ds(pp,s,nbhd,1,2)
	a = get_idm_dv(cs[1].behavior.p_idm,pp.dt,v,dv,ds)/pp.dt
	println("Fast going slow behind slow: v=$v,dv=$dv,s=$ds,a=$a")
	cs = CarState[CarState(1.,2,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1),4),CarState(36.,2,27.,0,IDMMOBILBehavior("cautious",27.,4.,1),2)]
	s = MLState(false,false,0.0,0.0,cs)
	dmodel = IDMMOBILModel(length(cs),pp)
	nbhd = get_neighborhood(pp,s,1)
	v = cs[1].vel
	dv, ds = get_dv_ds(pp,s,nbhd,1,2)
	a = get_idm_dv(cs[1].behavior.p_idm,pp.dt,v,dv,ds)/pp.dt
	println("Fast going fast behind slow: v=$v,dv=$dv,s=$ds,a=$a")
	cs = CarState[CarState(1.,2,35.,0,IDMMOBILBehavior("cautious",27.,4.,1),6),CarState(36.,2,27.,0,IDMMOBILBehavior("cautious",27.,4.,1),3)]
	s = MLState(false,false,0.0,0.0,cs)
	dmodel = IDMMOBILModel(length(cs),pp)
	nbhd = get_neighborhood(pp,s,1)
	v = cs[1].vel
	dv, ds = get_dv_ds(pp,s,nbhd,1,2)
	a = get_idm_dv(cs[1].behavior.p_idm,pp.dt,v,dv,ds)/pp.dt
	println("slow going fast behind slow: v=$v,dv=$dv,s=$ds,a=$a")
	cs = CarState[CarState(1.,2,27.,0,IDMMOBILBehavior("cautious",27.,4.,1),7),CarState(36.,2,27.,0,IDMMOBILBehavior("cautious",27.,4.,1),8)]
	s = MLState(false,false,0.0,0.0,cs)
	dmodel = IDMMOBILModel(length(cs),pp)
	nbhd = get_neighborhood(pp,s,1)
	v = cs[1].vel
	dv, ds = get_dv_ds(pp,s,nbhd,1,2)
	a = get_idm_dv(cs[1].behavior.p_idm,pp.dt,v,dv,ds)/pp.dt
	println("slow going slow behind slow: v=$v,dv=$dv,s=$ds,a=$a")
	cs = CarState[CarState(1.,2,35.,0,IDMMOBILBehavior("cautious",27.,4.,1),9)]
	s = MLState(false,false,0.0,0.0,cs)
	dmodel = IDMMOBILModel(length(cs),pp)
	nbhd = get_neighborhood(pp,s,1)
	v = cs[1].vel
	dv, ds = get_dv_ds(pp,s,nbhd,1,2)
	a = get_idm_dv(cs[1].behavior.p_idm,pp.dt,v,dv,ds)/pp.dt
	println("slow going fast: v=$v,dv=$dv,s=$ds,a=$a")
	cs = CarState[CarState(1.,2,27.,0,IDMMOBILBehavior("cautious",35.,4.,1),10)]
	s = MLState(false,false,0.0,0.0,cs)
	dmodel = IDMMOBILModel(length(cs),pp)
	nbhd = get_neighborhood(pp,s,1)
	v = cs[1].vel
	dv, ds = get_dv_ds(pp,s,nbhd,1,2)
	a = get_idm_dv(cs[1].behavior.p_idm,pp.dt,v,dv,ds)/pp.dt
	println("fast going slow: v=$v,dv=$dv,s=$ds,a=$a")
	cs = CarState[CarState(1.,2,31.,0,IDMMOBILBehavior("cautious",27.,4.,1),11)]
	s = MLState(false,false,0.0,0.0,cs)
	dmodel = IDMMOBILModel(length(cs),pp)
	nbhd = get_neighborhood(pp,s,1)
	v = cs[1].vel
	dv, ds = get_dv_ds(pp,s,nbhd,1,2)
	a = get_idm_dv(cs[1].behavior.p_idm,pp.dt,v,dv,ds)/pp.dt
	println("slow going med: v=$v,dv=$dv,s=$ds,a=$a")
	cs = CarState[CarState(1.,2,31.,0,IDMMOBILBehavior("cautious",35.,4.,1),12)]
	s = MLState(false,false,0.0,0.0,cs)
	dmodel = IDMMOBILModel(length(cs),pp)
	nbhd = get_neighborhood(pp,s,1)
	v = cs[1].vel
	dv, ds = get_dv_ds(pp,s,nbhd,1,2)
	a = get_idm_dv(cs[1].behavior.p_idm,pp.dt,v,dv,ds)/pp.dt
	println("fast going med: v=$v,dv=$dv,s=$ds,a=$a")
end

function test_idm()
	println("\tTesting IDM Units...")
	test_idm_creation()
	test_idm_hashing()
	test_get_dv_ds()
	test_idm_dv()
end
