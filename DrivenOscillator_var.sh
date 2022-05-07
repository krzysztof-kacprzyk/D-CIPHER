for noise_ratio in 0.005 0.2 0.5 0.75 1.0
do
    echo "Noise ratio: $noise_ratio"
    nohup python -m var_objective.run_var_square DrivenHarmonicOscillator 0 2.0 20 $noise_ratio 200 NumbersRandom2 2spline1Dtrans 10 50 l1 lars-imp --seed 3 --num_samples 10;
done
