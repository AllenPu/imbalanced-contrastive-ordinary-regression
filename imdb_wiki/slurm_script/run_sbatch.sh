for i in 0.0001 0.0002; do
    for g in 10 20 25; do
            for e in 100 150 200; do
                for temp in 1 3 5 10; do
                    jobs='lr'_${i}_'tau_0.5'_'group'_${g}_'epoch'_${e}_'temp'_${temp}
                    echo ${jobs}
                    sbatch --job-name=${jobs} run_train.sh ${i} ${g} ${e} ${temp}
                done
            done
    done
done
