for i in `seq 0 874`; do #500

	qsub -pe omp 1 -l h_rt=140:00:00 -V ./run_one_cell.sh $i

done
