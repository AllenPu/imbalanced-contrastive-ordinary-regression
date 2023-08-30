for i in 0.0001 0.00005 0.0002; do
    for g in 10 20 25; do
        for e in 200 250; do
            for sigma in 0.5 1 1.5;do
                for contra in False; do
                    if [${contra} = True];then
                        for temp in 0.07 0.1 0.5; do
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
