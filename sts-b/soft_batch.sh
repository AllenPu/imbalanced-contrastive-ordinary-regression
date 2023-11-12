for i in 0.0001 0.0005; do
    for g in 10 20; do
        for p in 200 300; do
            for temp in 0.02 0.05 0.07; do
                for sigma in 0.7 1 1.5; do
                    jobs='sts'_${i}_'group'_${g}_'epoch'_${e}_'temp'_${temp}
                    echo ${jobs}
                    sbatch --job-name=${jobs} ./slurm_jobs/soft.sh ${i} ${g} ${p} ${temp} ${sigma}
                done
            done
        done
    done
done