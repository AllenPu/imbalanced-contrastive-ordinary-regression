for i in 0.0002 0.0003 0.00008; do
    for g in 5 10 20; do
        for e in 450; do
            for sigma in 1 2 5 7;do
                for contra in True; do
                    if [ $contra ];then
                        for temp in 0.02 0.03 0.04; do
                            jobs='contra_'+'lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}_'temp'_${temp}
                            echo ${jobs}
                            sbatch --job-name=${jobs} ./slurm_jobs/train_contra.sh ${i} ${g} ${e} ${temp} ${sigma}
                        done
                    else
                        jobs='lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}
                        echo ${jobs}
                        sbatch --job-name=${jobs} ./slurm_jobs/train.sh ${i} ${g} ${e} ${sigma}
                    fi 
                done
            done
        done
    done
done
