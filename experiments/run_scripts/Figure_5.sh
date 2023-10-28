echo "Running Figure 5 experiments..."
for noise_ratio in 0.001 0.01 0.015
do
    echo "Noise ratio: $noise_ratio"

    nohup python -m var_objective.run_var_square WaveEquation3_L1 0 2.0 30 $noise_ratio 200 HeatRandom 2spline2Dtrans 10 0 l1 lars-imp --seed 437782 --num_samples 10;

    nohup python -m var_objective.run_mse_square WaveEquation3_L1 0 2.0 30 $noise_ratio HeatRandom gp 0 l1 lars-imp --seed 437782 --num_samples 10;
done

echo "Figure 5 experiments complete."