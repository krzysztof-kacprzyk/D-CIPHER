for noise_ratio in 0.0001 0.001 0.01 0.1
do
    echo "Noise ratio: $noise_ratio"
    nohup python -m var_objective.run_mse_square WaveEquation1_L1 0 2.0 30 $noise_ratio HeatRandom gp 10 l1 lars-imp --seed 2 --num_samples 10;
done
