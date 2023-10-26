for i in 0.009 0.008; do
    for g in 10 25 50; do
        for e in 300; do
            for sigma in 2.1 2.5;do
                for contra in True; do
                    if [ $contra ];then
                        for temp in 0.02 0.05 0.07 0.09; do
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
