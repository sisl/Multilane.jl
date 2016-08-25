function sbatch_spawn(tests::AbstractVector, objects::Dict;
                      batch_size=100,
                      time_per_batch="20:00",
                      data_dir=joinpath(get(ENV, "SCRATCH", tempdir()),
                                        string("sim_data_", Dates.format(Dates.now(),"u_d_HH_MM"))),
                      submit_command="sbatch",
                      template_name="sherlock.sh",
                      job_name=string("Multilane_", randstring())
                      )

    stats = setup_stats(tests, objects)
    # shuffle so that batches take approximately the same amount of time
    sort!(stats, cols=:uuid)
    objects["tests"] = Dict([t.key=>t for t in tests])

    try
        mkdir(data_dir)
    catch ex
        warn("When creating data_dir, got error $ex. Does the dir already exist?\nContinuing anyways.")
    end
    objectname = joinpath(data_dir, string("objects_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld"))
    JLD.save(objectname, objects)
    println("objects saved to $objectname")

    nb_sims = nrow(stats)
    nb_batches = cld(nb_sims, batch_size)
    @assert rem(nb_sims, batch_size) == 0
    results_file_list = []

    listname = joinpath(data_dir, string("list_\${SLURM_ARRAY_TASK_ID}_of_$(nb_batches).jld"))

    tpl = Mustache.parse(readall(joinpath(Pkg.dir("Multilane"), "templates", template_name)))
    sbatch = Mustache.render(tpl,
                    job_name=job_name,
                    time=time_per_batch,
                    object_file_path=objectname,
                    list_file_path=listname,
                    nb_batches=nb_batches,
                    data_dir=data_dir)

    sbatchname = joinpath(data_dir, "$job_name.sbatch")
    open(sbatchname, "w") do f
        write(f, sbatch)
    end

    for i in 1:nb_batches
        println("preparing job $i of $nb_batches")

        these_stats = stats[(i-1)*batch_size+1:i*batch_size, :]
        
        this_listname = replace(listname, "\${SLURM_ARRAY_TASK_ID}", i)
        JLD.save(this_listname, Dict("stats"=>these_stats))
        
        push!(results_file_list, joinpath(data_dir, string("results_$(i)_of_$(nb_batches).jld")))
    end

    cmd = `$submit_command $sbatchname` 
    println("running $cmd ...")
    @time run(cmd)
    println("done")

    results_list_file = joinpath(data_dir, "results_list.txt")
    println("saving results file list to")
    println(results_list_file)
    open(results_list_file, "w") do opened
        for f in results_file_list
            write(opened, "$f\n")
        end
    end

    return results_file_list
end

function gather_results(results_file_list; save_file::Nullable=Nullable())
    results = JLD.load(first(results_file_list))
    # stats = results["stats"]
    for (i,f) in enumerate(results_file_list[2:end])
        print("\rmerging file $(i+1) of $(length(results_file_list))")
        results = merge_results!(results, JLD.load(f))
        # new_stats = JLD.load(f,"stats")
        # stats = append!(stats, new_stats)
    end
    println()
    # results["stats"] = stats
    if !isnull(save_file)
        JLD.save(get(save_file), results)
    end
    return results
end
