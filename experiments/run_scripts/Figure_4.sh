echo "Running Figure 4 experiments..."

# D-CIPHER
echo "D-CIPHER"
for noise_ratio in 0.005 0.2 0.5 0.75 1.0
do
    echo "Noise ratio: $noise_ratio"
    nohup python -m var_objective.run_var_square DrivenHarmonicOscillator 0 2.0 20 $noise_ratio 200 NumbersRandom2 2spline1Dtrans 10 50 l1 lars-imp --seed 3 --num_samples 10;
done

for freq in 5 10 15 25
do
    echo "Freq: $freq"
    nohup python -m var_objective.run_var_square DrivenHarmonicOscillator 0 2.0 $freq 0.01 200 NumbersRandom2 2spline1Dtrans 10 50 l1 lars-imp --seed 2 --num_samples 10;
done

for num_samples in 1 2 5 10 15
do
    echo "Num samples: $num_samples"
    nohup python -m var_objective.run_var_square DrivenHarmonicOscillator 0 2.0 20 0.01 200 NumbersRandom2 2spline1Dtrans 10 50 l1 lars-imp --seed 2 --num_samples $num_samples;
done

# Ablated D-CIPHER
echo "Ablated D-CIPHER"
for noise_ratio in 0.005 0.2 0.5 0.75 1.0
do
    echo "Noise ratio: $noise_ratio"
    nohup python -m var_objective.run_mse_square DrivenHarmonicOscillator 0 2.0 20 $noise_ratio NumbersRandom2 gp 50 l1 lars-imp --seed 3 --num_samples 10;
done

for freq in 5 10 15 25
do
    echo "Freq: $freq"
    nohup python -m var_objective.run_mse_square DrivenHarmonicOscillator 0 2.0 $freq 0.01 NumbersRandom2 gp 50 l1 lars-imp --seed 2 --num_samples 10;
done

for num_samples in 1 2 5 10 15
do
    echo "Num samples: $num_samples"
    nohup python -m var_objective.run_mse_square DrivenHarmonicOscillator 0 2.0 20 0.01 NumbersRandom2 gp 50 l1 lars-imp --seed 2 --num_samples $num_samples;
done

echo "Figure 4 experiments complete."
