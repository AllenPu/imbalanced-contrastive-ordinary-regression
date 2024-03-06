for i in 0.0001 0.0005 0.00001; do
    #jobs='lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}_'temp'_${temp}
    #echo ${jobs}
    sbatch --job-name=${jobs} ./slurm_jobs/train.sh ${i} 
done
