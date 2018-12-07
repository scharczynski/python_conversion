for i in `seq 0 100`; do 

	qsub -pe omp 1 -l h_rt=140:00:00 -V ./run_one_cell.sh $0

done