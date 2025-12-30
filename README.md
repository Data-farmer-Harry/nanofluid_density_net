# density_net

## Overview
This project trains ML models to predict the `c_myRDF[2]` value from RDF-derived features.
It supports a feed-forward neural network and an XGBoost regressor, plus scripts for
evaluating errors and plotting results.


![image][(https://github.com/MaiEmily/map/blob/master/public/image/20190528145810708.png](https://github.com/Data-farmer-Harry/nanofluid_density_net/blob/main/figure/width_45.0_temp_314.0.png))


## Data format
The training/test data are stored in `data/*.npz` with a single array key `data`.
Each row is a single sample with 7 columns in this order:

1. `Row` (r)
2. `width`
3. `temperature`
7. label: `c_myRDF[2]`

The script `sample.py` builds this format from raw `rdf_static_result_*.txt` files
and writes `data/train.npz`, `data/valid.npz`, and `data/test.npz`.

## Project structure
- `model.py` - defines the neural network `densityNet`, dataset wrapper, and helpers.
- `train_nn.py` - trains the NN once at a fixed learning rate and saves `train/NN/model.pth`.
- `nn_training_mse.py` / `train copy 2.py` - sweep learning rates and log test MSE to `nnerro.xlsx`.
- `train_xgboost.py` - XGBoost training with randomized hyperparameter search.
- `plot.py` - compare MD, XGBoost, and NN predictions for each (width, temperature).
- `plot_error.py` - XGBoost prediction with absolute error subplot.
- `nine_picture.py` - 3x3 panel plot for selected widths/temperatures + error bars.
- `nn_learning rate_plot.py` - plot learning rate vs test metric from `nnerro.xlsx`.
- `data/` - `train.npz`, `valid.npz`, `test.npz`.
- `train/NN/` - saved NN model (`model.pth`).
- `train/XGBoost/` - saved XGBoost model (`xgb_model.json`).
- `figure/` - output plots.

## Requirements
Core ML and plotting:
- Python 3.x
- numpy, pandas, matplotlib
- torch
- xgboost
- scikit-learn

Optional for data preparation:
- mdtraj
- scipy

## Quick start
1. Prepare data (from RDF text files):
   - Put raw files under `use_data/rdf_results_width_*/rdf_static_result_*_*.txt`
   - Run:
     ```bash
     python sample.py
     ```
   - This writes `data/train.npz`, `data/valid.npz`, `data/test.npz`.

2. Train NN (single run):
   ```bash
   python train_nn.py
   ```
   Output: `train/NN/model.pth`

3. Train XGBoost:
   ```bash
   python train_xgboost.py
   ```
   Output: `train/XGBoost/xgb_model.json`

4. Learning-rate sweep (MSE):
   ```bash
   python nn_training_mse.py
   ```
   Output: `nnerro.xlsx` with `learning_rate` and `test_mse`.

5. Plot learning rate curve:
   ```bash
   python "nn_learning rate_plot.py"
   ```
   Output: `figure/nn_learning_rate_plot.png`

6. Plot predictions and errors:
   ```bash
   python plot.py
   python plot_error.py
   python nine_picture.py
   ```
   Outputs under `figure/`.

## Notes
- Normalization is computed from `data/train.npz` and used consistently for NN inference.
- The NN uses MSE loss for training; evaluation metrics may differ by script.
- `train_xgboost.py` uses MAE as the scoring metric for hyperparameter search.

## Outputs
- `nnerro.xlsx` - learning rate sweep results.
- `figure/*.png` - model predictions and error visualizations.
- `train/NN/model.pth` and `train/XGBoost/xgb_model.json` - trained model files.
