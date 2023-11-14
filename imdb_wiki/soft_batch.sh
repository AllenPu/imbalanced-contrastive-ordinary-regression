for i in 0.00005; do
    for g in 20; do
        for e in 90; do
            for sigma in 1 1.5 2; do
                for contra in True; do
                    if [ $contra ];then
                        for temp in 0.02 0.05; do
                            jobs='imdb_wiki_lr'_${i}_'tau_1'_'group'_${g}_'epoch'_${e}_'temp'_${temp}
                            echo ${jobs}
                            sbatch --job-name=${jobs} ./slurm_script/soft.sh ${i} ${g} ${e} ${temp} ${sigma}
                        done
                    else
                        jobs='imdb_wiki_lr_lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}
                        echo ${jobs}
                        sbatch --job-name=${jobs} ./slurm_script/train.sh ${i} ${g} ${e} ${sigma}
                    fi
                done
            done
        done
    done
done