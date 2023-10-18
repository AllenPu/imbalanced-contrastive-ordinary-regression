for i in 0.0001 0.0005; do
    for g in 10 20; do
        for e in 200 300; do
            for sigma in 1 1.5 2;do
                for contra in True; do
                    if [ $contra ];then
                        for temp in 0.02 0.03 0.05; do
                            jobs='contra_'+'lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}_'temp'_${temp}
                            echo ${jobs}
                            sbatch --job-name=${jobs} ./slurm_jobs/uncertain.sh ${i} ${g} ${e} ${temp} ${sigma}
                        done
                    else
                        jobs='lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}
                        echo ${jobs}
                        #sbatch --job-name=${jobs} ./slurm_jobs/train.sh ${i} ${g} ${e} ${sigma}
                    fi 
                done
            done
        done
    done
done
