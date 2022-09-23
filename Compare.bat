python -m var_objective.compare_with_sindy HeatEquationHomo 0 2.0 30 0.1 200 HeatRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10

python -m var_objective.compare_with_sindy BurgerDict 0 2.0 20 0.001 200 BurgerRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10

python -m var_objective.compare_with_sindy BurgerDict 0 2.0 20 0.01 200 BurgerRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10

python -m var_objective.compare_with_sindy BurgerDict 0 2.0 20 0.1 200 BurgerRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10

python -m var_objective.run_var_square_dict SLM1Dict 0 1.0 20 0.001 100 PopulationRandom 2spline2Dtrans 5 10 l1 lars-imp --seed 2 --num_samples 10

python -m var_objective.run_var_square_dict SLM1Dict 0 1.0 20 0.01 100 PopulationRandom 2spline2Dtrans 5 10 l1 lars-imp --seed 2 --num_samples 10