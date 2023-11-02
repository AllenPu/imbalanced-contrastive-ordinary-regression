for i in 0.0001 0.0005; do
    for g in 10; do
        for e in 90; do
            for temp in 0.02 0.05; do
                for sigma in 1 1.5 2; do
                    jobs='imdb_wiki_lr'_${i}_'tau_1'_'group'_${g}_'epoch'_${e}_'temp'_${temp}
                    echo ${jobs}
                    sbatch --job-name=${jobs} ./slurm_script/soft.sh ${i} ${g} ${e} ${temp} ${sigma}
                done
            done
        done
    done
done