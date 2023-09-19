for i in 1e-3 5e-3 5e-4 1e-4; do
    for g in 5 10 20; do
        for s in 1 2; do
            for e in 100 200 300:do
                for la in True False; do
                    if [ $la ];then
                        jobs='la_true_contra_'+'lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}
                        echo ${jobs}
                        sbatch --job-name=${jobs} ./slurm_jobs/run_la.sh ${i} ${g} ${e} ${sigma}
                        
                    else
                        jobs='la_false_lr'_${i}_'groups'_${g}_'sigma'_${s}_'epoch'_${e}
                        echo ${jobs}
                        sbatch --job-name=${jobs} ./slurm_jobs/run.sh ${i} ${g} ${s} ${e}
                done
            done
        done
    done
done