using PGFPlots
using CSV
using DataFrames
using Missings
using Query

filename = Pkg.dir("Multilane", "data/baseline_curve_Tuesday_20_Mar_23_20.csv")
data = CSV.read(filename, nullable=true)

# XXX ILLEGALLY REPLACING DATA
data[ismissing.(data[:steps_to_lane]), :steps_to_lane] = 1000
data[:time_to_lane] = data[:steps_to_lane].*data[:dt]
data[:brakes_per_step] = data[:nb_brakes]./data[:n_steps]

combined = by(data, :lambda) do df
    return DataFrame(ttl=mean(df[:time_to_lane]),
                     ttl_sem=std(df[:time_to_lane])./sqrt(size(df,1)),
                     bps=mean(df[:brakes_per_step]),
                     bps_sem=std(df[:brakes_per_step])./sqrt(size(df,1)),
                     nb_missing=sum(df[:steps_to_lane].==1000)
                    )
end

# selected = @from r in data begin
#     @where r.lambda == 2
#     @select r.time_to_lane
#     @collect
# end
