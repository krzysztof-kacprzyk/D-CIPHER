python -m var_objective.compare_with_sindy HeatEquationHomo 0 2.0 30 0.001 200 HeatRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --sign_index 1 --sindy_order 2
python -m var_objective.compare_with_sindy HeatEquationHomo 0 2.0 30 0.01 200 HeatRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --sign_index 1 --sindy_order 2
python -m var_objective.compare_with_sindy HeatEquationHomo 0 2.0 30 0.1 200 HeatRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --sign_index 1 --sindy_order 2

python -m var_objective.compare_with_sindy BurgerDict 0 2.0 20 0.001 200 BurgerRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --sign_index 1 --sindy_order 2
python -m var_objective.compare_with_sindy BurgerDict 0 2.0 20 0.01 200 BurgerRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --sign_index 1 --sindy_order 2
python -m var_objective.compare_with_sindy BurgerDict 0 2.0 20 0.1 200 BurgerRandom 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --sign_index 1 --sindy_order 2

python -m var_objective.compare_with_sindy KSDict 0 100.0 60 0.001 200 HeatRandom 4spline2Dtrans 10 5 l1 lars-imp --seed 2 --num_samples 1 --sign_index 1 --sindy_order 4
python -m var_objective.compare_with_sindy KSDict 0 100.0 60 0.01 200 HeatRandom 4spline2Dtrans 10 5 l1 lars-imp --seed 2 --num_samples 1 --sign_index 1 --sindy_order 4
python -m var_objective.compare_with_sindy KSDict 0 100.0 60 0.1 200 HeatRandom 4spline2Dtrans 10 5 l1 lars-imp --seed 2 --num_samples 1 --sign_index 1 --sindy_order 4

