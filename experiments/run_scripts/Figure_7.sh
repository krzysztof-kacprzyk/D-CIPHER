echo "Running Figure 7 experiments..."
for eq_number in 0 1 2 3
do
    echo "Equation number: $eq_number"

    # If eq_number is 0 or 1, then sign_index is 2. Otherwise, it is 0.

    if [ $eq_number -lt 2 ]
    then
        sign_index=2
    else
        sign_index=0
    fi

    for noise_ratio in 0.001 0.01 0.1
    do
        echo "Noise ratio: $noise_ratio"
        
        python -m var_objective.run_var_square_dict FullFlow2D $eq_number 2.0 20 $noise_ratio 200 SourcesRandom2D 2spline2Dtrans 10 10 l1 lars-imp --seed 2 --num_samples 10 --no_gp --sign_index $sign_index --exp_name CauchyRiemann;
    done
done

echo "Figure 7 experiments complete."