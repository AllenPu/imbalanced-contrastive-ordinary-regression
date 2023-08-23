for i in 1e-4 1e-3 5e-5 1e-5; do
    for g in 5 10 20 25; do
        jobs='lr'_${i}_'group'_${g}
        echo ${jobs}
        sbatch --job-name=${jobs} train.sh ${i} ${g}
    done
done