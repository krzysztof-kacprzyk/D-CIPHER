echo "Running Table 3 experiments..."

# D-CIPHER
echo "D-CIPHER"
for noise_ratio in 0.05 0.1 0.2
do
    echo "Noise ratio: $noise_ratio"
    python -m var_objective.run_var_square HeatEquation5_L1 0 2.0 30 $noise_ratio 200 HeatRandom 2spline2Dtrans 10 50 l1 lars-imp --seed 2 --num_samples 10 --exp_name HeatSource;
done

# Ablated D-CIPHER
echo "Ablated D-CIPHER"
for noise_ratio in 0.05 0.1 0.2
do
    echo "Noise ratio: $noise_ratio"
    python -m var_objective.run_mse_square HeatEquation5_L1 0 2.0 30 $noise_ratio HeatRandom gp 50 l1 lars-imp --seed 2 --num_samples 10 --exp_name HeatSource;
done

echo "Table 3 experiments complete."