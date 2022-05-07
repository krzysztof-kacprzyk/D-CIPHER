for num_samples in 1 2 5 10 15
do
    echo "Num samples: $num_samples"
    nohup python -m var_objective.run_mse_square DrivenHarmonicOscillator 0 2.0 20 0.01 NumbersRandom2 gp 50 l1 lars-imp --seed 2 --num_samples $num_samples;
done
