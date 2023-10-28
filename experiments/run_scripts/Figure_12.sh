echo "Running Figure 12 experiments..."

for noise_ratio in 0.001 0.01 0.1
do
    echo "Noise ratio: $noise_ratio"

    nohup python -m var_objective.run_var_square_dict_interpolation KSDict 0 100.0 60 $noise_ratio 200 HeatRandom 4spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 1 --sign_index 1 --no_gp;
    nohup python -m var_objective.run_var_square_dict_interpolation HeatEquationHomo 0 2.0 30 $noise_ratio 200 HeatRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --sign_index 1 --no_gp;
    nohup python -m var_objective.run_var_square_dict_interpolation BurgerDict 0 2.0 20 $noise_ratio 200 BurgerRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --sign_index 1 --no_gp;
done

echo "Figure 12 experiments complete."