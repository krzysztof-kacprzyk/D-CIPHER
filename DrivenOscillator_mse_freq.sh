for freq in 5 10 15 25
do
    echo "Freq: $freq"
    nohup python -m var_objective.run_mse_square DrivenHarmonicOscillator 0 2.0 $freq 0.01 NumbersRandom2 gp 50 l1 lars-imp --seed 2 --num_samples 10;
done
