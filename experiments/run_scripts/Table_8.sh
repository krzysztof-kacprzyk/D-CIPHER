echo "Running Table 8 experiments..."

# D-CIPHER
echo "D-CIPHER"
for noise_ratio in 0.001 0.01
do
    echo "Noise ratio: $noise_ratio"
    python -m var_objective.run_var_square_dict SLM1Dict 0 1.0 20 $noise_ratio 100 PopulationRandom 2spline2Dtrans 5 10 l1 lars-imp --seed 2 --num_samples 10 --exp_name SLM;
done


# Ablated D-CIPHER
echo "Ablated D-CIPHER"
for noise_ratio in 0.001 0.01
do
    echo "Noise ratio: $noise_ratio"
    python -m var_objective.run_mse_square_dict SLM1Dict 0 1.0 20 $noise_ratio PopulationRandom gp 10 l1 lars-imp --seed 2 --num_samples 10 --exp_name SLM;
done

echo "Table 8 experiments complete."