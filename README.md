### Machine-Learning-Assignment
# BLEVE Blast Pressure Prediction
Predicting peak overpressure from Boiling Liquid Expanding Vapour Explosions (BLEVEs) using machine learning. Built as part of COMP3010 Machine Learning at Curtin University.
Kaggle Private Leaderboard MAPE: 0.1926 (top scoring bracket <0.20)

# Problem
BLEVEs occur when pressurised LPG tanks fail catastrophically, generating blast waves that pose serious risks to nearby structures and people. Predicting the peak pressure at different points around a nearby obstacle is difficult using conventional physics-based methods due to the complex, nonlinear nature of the explosion dynamics.
This project uses a data-driven ML approach to predict peak pressure readings (in bar) across 27 sensors placed on the front, back, and side walls of a rigid obstacle, given physical variables describing the tank, explosion conditions, and sensor location.

# Dataset
- Train set: ~10,000 rows, 25 features including target pressure
- Test set: ~3,200 rows, 24 features (no target)
- Features include: tank dimensions, failure pressure, liquid/vapour temperatures, obstacle geometry, sensor 3D coordinates, liquid state (subcooled/superheated)
- Target: Peak pressure at each sensor location (bar)

## Approach
# Data Preprocessing
- Dropped rows with more than 5 missing values (10 rows)
- Imputed remaining numeric missing values with the column median; categorical with mode
- Capped Tank Failure Pressure at the 99th percentile to remove corrupted entries
- Standardised 9 inconsistent string variants in the Status column (e.g. Saperheated, Subcool) to two clean categories
- Removed 50 duplicate rows
- Log-transformed the target variable to handle right-skewed pressure distribution

# Feature Engineering
Four new features derived from existing columns:
- Tank Volume = Width × Length × Height
- Sensor Distance to BLEVE = Euclidean distance from sensor to origin (√x² + y² + z²)
- Obstacle Volume = Width × Height × Thickness
- Obstacle Face Area = Width × Height
Cross-validation confirmed these features improved R² from 0.8735 to 0.8876.

# Feature Selection
Used Pearson correlation and Random Forest feature importances to identify and drop 4 low-signal features (Status, Liquid Critical Pressure, Liquid Boiling Temperature, Liquid Critical Temperature), with no loss in CV performance.

# Models Compared
ModelR² (val)MAPE (val)RMSE (val)Ridge 
Regression 0.77 0.41 — 
Random Forest 0.96 0.16 —
Neural Network (MLP) 0.97 0.13 —

# Hyperparameter Tuning
- Ridge: Grid search over alpha [0.01, 0.1, 1, 10, 100, 1000]
- Random Forest: Randomised search (30 iterations) over n_estimators, max_depth, min_samples_leaf, min_samples_split, max_features
- MLP: Randomised search (20 iterations) over hidden_layer_sizes, alpha, learning_rate_init

# Final Model
Neural Network (MLP) — selected for lowest validation MAPE (0.13). Retrained on the full training set before generating test predictions. Log-transformed predictions converted back to bar scale via exp().

# Results
Set MAPE R² 
Validation 0.13 0.97 
Kaggle Private Test 0.1926 —

# Tech Stack
- Python, Jupyter Notebook (Google Colab)
- pandas, NumPy
- scikit-learn (Ridge, RandomForestRegressor, MLPRegressor, GridSearchCV, RandomizedSearchCV)
- Kaggle competition submission






