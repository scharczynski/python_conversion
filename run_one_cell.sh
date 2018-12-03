#!/bin/bash -l

#module load mcr/9.0.1_2016a 
#./stim_spec_time_cells_16a $1

module load python3

echo pwd

echo $1

python3 run_cell.py $1 $1
