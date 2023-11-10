echo "Running Figure 3 experiments..."
for noise_ratio in 0.001 0.005 0.01 0.05 0.1 0.2
do
    echo "Noise ratio: $noise_ratio"

    python -m var_objective.compare_with_sindy_more HeatEquationHomo 0 2.0 30 $noise_ratio 200 HeatRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --sign_index 1 --sindy_order 2 --exp_name HeatHomogeneous;

    python -m var_objective.compare_with_sindy_more BurgerDict 0 2.0 20 $noise_ratio 200 BurgerRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --sign_index 1 --sindy_order 2 --exp_name Burger;

    python -m var_objective.compare_with_sindy_more KSDict 0 100.0 60 $noise_ratio 200 HeatRandom 4spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 1 --sign_index 1 --sindy_order 4 --exp_name KS;
done

echo "Figure 3 experiments complete."





