#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name={{{:job_name}}}
#################  
#dir for output
#SBATCH --workdir={{{:data_dir}}}
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time={{{:time}}}
#################
#number of jobs in array
#SBATCH --array=1-{{{:nb_batches}}}
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#tasks to run per node; a "task" is usually mapped to a MPI processes.
# for local parallelism (OpenMP or threads), use "--ntasks-per-node=1 --cpus-per-task=16" instead
#SBATCH --ntasks-per-node=1
#################

~/bin/julia ~/.julia/v0.5/Multilane/scripts/runsims.jl {{{:object_file_path}}} {{{:list_file_path}}}
