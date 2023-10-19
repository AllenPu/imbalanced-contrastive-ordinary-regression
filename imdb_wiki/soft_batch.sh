for i in 0.0001 0.0005 0.00001; do
    for g in 10 20 ; do
        for e in 200 250; do
            for temp in 0.02 0.05; do
                for sgima in 1 1.5 2; do
                    jobs='lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}_'temp'_${temp}
                    echo ${jobs}
                    sbatch --job-name=${jobs} ./slurm_jobs/soft.sh ${i} ${g} ${e} ${temp} ${sigma}
                done
            done
        done
    done
done