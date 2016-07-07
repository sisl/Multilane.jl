# Multilane

[![Build Status](https://travis-ci.org/zsunberg/Multilane.jl.svg?branch=master)](https://travis-ci.org/zsunberg/Multilane.jl)

Simulation and control of an autonomous car on a multilane highway in a (PO)MDP framework (implemented using [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl)).

![Simulation Image](https://raw.githubusercontent.com/zsunberg/Multilane.jl/master/img/env.png)

The following solvers can be used:
- [MCTS](https://github.com/JuliaPOMDP/MCTS.jl)
- [POMCP](https://github.com/JuliaPOMDP/POMCP.jl)
- [RobustMCTS](https://github.com/zsunberg/RobustMCTS.jl/tree/master/src)
