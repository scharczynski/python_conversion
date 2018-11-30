for i in `seq 1 500`; do #500
#cells="288 332 335 373 159 291 190 158 232 324 57 3 378 405 40 46 58 82 399"
#cells="365 274 241 209 152 136 70 40"
#cells="135"
#for i in `echo $cells`; do 
	#for j in `seq 1 20`; do	
		#qsub -q tcn -pe omp 1 -l h_rt=24:00:00 -V ./run_one_job.sh $i
		qsub -pe omp 1 -l h_rt=140:00:00 -V ./run_one_job.sh $i
		#qsub -pe omp 1 -V ./run_one_job.sh $i
	#if [ i % 5 = 0 ]; then
	#sleep 60s
	#fi		
	#done
done
