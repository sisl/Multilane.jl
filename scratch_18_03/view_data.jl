using CSV
using DataFrames
using Missings
using Query

filename = Pkg.dir("Multilane", "data/baseline_curve_Thursday_22_Mar_18_41.csv")
data = CSV.read(filename, nullable=true)

data[:time_to_lane] = data[:steps_to_lane].*data[:dt]
data[:brakes_per_step] = data[:nb_brakes]./data[:n_steps]

combined = by(data, [:solver, :lambda]) do df
    n = size(df, 1)
    return DataFrame(reached_lane=sum(df[:terminal].=="lane")/n,
                     had_brakes=sum(df[:nb_brakes].>=1)/n,
                    )

    # return DataFrame(ttl=mean(df[:time_to_lane]),
    #                  ttl_sem=std(df[:time_to_lane])./sqrt(size(df,1)),
    #                  bps=mean(df[:brakes_per_step]),
    #                  bps_sem=std(df[:brakes_per_step])./sqrt(size(df,1)),
    #                  nb_missing=sum(df[:steps_to_lane].==1000)
    #                 )
end

@show combined

# selected = @from r in data begin
#     @where r.lambda == 2
#     @select r.time_to_lane
#     @collect
# end
