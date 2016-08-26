#!/bin/bash 

for SLURM_ARRAY_TASK_ID in `seq 1 {{{:nb_batches}}}`; do
    julia ~/.julia/Multilane/scripts/runsims.jl {{{:object_file_path}}} {{{:list_file_path}}} &
done
