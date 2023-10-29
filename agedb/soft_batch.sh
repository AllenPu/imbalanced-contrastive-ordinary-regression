for i in 0.001 0.0015; do
    for g in 25; do
        for e in 300; do
            for sigma in 1 1.5 2 2.5;do
                for contra in True; do
                    if [ $contra ];then
                        for temp in 0.02 0.05 0.07 0.1; do
                            jobs='agedb_contra_lr'_${i}_'tau_1'_'group'_${g}_'epoch'_${e}_'temp'_${temp}
                            echo ${jobs}
                            sbatch --job-name=${jobs} ./slurm_jobs/uncertain.sh ${i} ${g} ${e} ${temp} ${sigma}
                        done
                    else
                        jobs='agedb_lr'_${i}_'tau_1'_'group'_${g}_'epoch'_${e}
                        echo ${jobs}
                        #sbatch --job-name=${jobs} ./slurm_jobs/train.sh ${i} ${g} ${e} ${sigma}
                    fi 
                done
            done
        done
    done
done
