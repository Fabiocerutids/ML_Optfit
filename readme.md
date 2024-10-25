## Repository Description and Overview
The repository contains the code I developed to ease: Hyperparameter Tuning for Traditional ML, Neural Network Building and its Hyperparameter Optimization.

To automate hyperparameter search and NN Building, make sure to pass dictionaries with the structure provided in the tutorial folder.

Note: The repository exploits Optuna as the library of choice to perform hyperparameter search. To familiarize with the library: https://optuna.org/

## Installation Guide
To install the package, please perform the following steps:
1. Clone the repository:
```bash
git clone https://github.com/Fabiocerutids/ML_Optfit.git
```

2. Locate yourself in the ML_Optfit folder:
```bash
cd ML_Optfit/
```

3. Run the following command in terminal:
```bash
pip install .
```

4. ML_Optfit is now installed, verify by running the command below:
```python
from ml_optfit.ml_optfit import HyperOptimNN
```