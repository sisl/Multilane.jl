using POMDPs

POMDPs.add("GenerativeModels")
POMDPs.add("POMDPToolbox")

try
    Pkg.clone("https://github.com/slundberg/PmapProgressMeter.jl.git")
catch
    println("already installed.")
end

try
    Pkg.clone("https://github.com/JuliaPlots/StatPlots.jl.git")
catch
    println("already installed.")
end
