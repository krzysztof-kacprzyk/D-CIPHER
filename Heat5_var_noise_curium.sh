for noise_ratio in 0.01
do
    echo "Noise ratio: $noise_ratio"
    nohup python -m var_objective.run_var_square HeatEquation5_L1 0 2.0 30 $noise_ratio 200 HeatRandom 2spline2Dtrans 10 50 l1 lars-imp --seed 2 --num_samples 10;
done
