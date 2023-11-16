Code for paper "D-CIPHER: Discovery of Closed-form Partial Differential Equations"

# Installation

Clone this repository and all submodules. You can do it by using the following code:
```
git clone --recurse-submodules -j8 https://github.com/krzysztof-kacprzyk/D-CIPHER
```

Install all dependencies in environment.yml. You can do so using conda and the following code:
```
conda env create --file=environment.yml
```

You also have to install PyTDMA. To do so, execute the following code:
```
cd PyTDMA
python setup.py build_ext --inplace
```

# Replicating experiments

Shell scripts to replicate the experiments can be found in `experiments/run_scripts`. You can run experiments separately or run all of them using `run_all.sh` file. Note, some of the experiments take a long time to run (~ a few days). Please run all experiments from the root directory.

To reproduce all figures and tables, use Jupyter notebooks in `experiments/analysis`.