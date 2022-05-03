for noise_ratio in 0.0001 0.001 0.01 0.1
do
    echo "Noise ratio: $noise_ratio"
    nohup python -m var_objective.run_var_square DrivenHarmonicOscillator 0 2.0 20 $noise_ratio 200 NumbersRandom2 2spline1Dtrans 10 50 l1 lars-imp --seed 3 --num_samples 10;
done
