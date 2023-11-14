for i in 0.0012 0.0017; do
    for g in 20; do
        for e in 300; do
            for sigma in 1 2;do
                for contra in True; do
                    if [ $contra ];then
                        for temp in 0.02 0.07; do
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
