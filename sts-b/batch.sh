for i in 1e-3 5e-3 5e-4 1e-4; do
    for g in 5 10 20 25; do
        for s in 0.5 1 1.5 2; do
            for e in 100 200 300:do
                jobs='lr'_${i}_'groups'_${g}_'sigma'_${s}_'epoch'_${e}
                echo ${jobs}
                sbatch --job-name=${jobs} ./slurm_jobs/run.sh ${i} ${g} ${s} ${e}
            done
        done
    done
done