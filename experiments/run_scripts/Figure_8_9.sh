echo "Running Figure 8 and 9 experiments"

python -m var_objective.run_var_square_dict_many SLM1DictMany 0 1.0 20 0.001 100 PopulationRandom 2spline2Dtrans 5 10 l1 lars-imp --seed 2 --num_samples 10 --exp_name Figure_8_9;

echo "Finished Figure 8 and 9 experiments"