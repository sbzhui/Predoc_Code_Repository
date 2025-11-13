# Economics-Informed Machine Learning – Coding Sample

This repository contains a self-contained coding sample that I use for pre-doctoral applications.  
It demonstrates how to combine **economic structure (DSGE models)** with **modern machine learning methods** to forecast macroeconomic time series.

The core idea:

> Use a DSGE model to generate **simulated “source” data** and treat it as an informative prior or pretraining signal for:
> - a **DSGE-VAR baseline** (Bayesian VAR with simulation-based priors),
> - an **Adaptive Transfer Gaussian Process (ATGP)**,
> - and a **Neural Network transfer-learning model (MLP)**,

then compare their forecasting performance on **real-world macro data**.

---

## 1. Repository structure

**Main entry point**

| File            | Type     | Description |
|----------------|----------|-------------|
| `main.ipynb`   | Notebook | End-to-end workflow: data loading, preprocessing, model estimation (DSGE-VAR, ATGP, MLP transfer learning), evaluation, and plotting. This is the file I recommend opening first. |

**Python modules**

| File                | Role / Contents |
|---------------------|-----------------|
| `Preprocess_data.py`| Time-series preprocessing utilities. In particular: `get_lag(df, lags, columns)` adds lagged regressors for each macro variable and drops missing values. |
| `Sim_data_gen.py`   | DSGE data generator. Draws economically plausible parameters for a small DSGE (New-Keynesian-style) model, simulates macro variables (e.g., output, inflation, interest rate), transforms them to growth/annualized rates, and writes each simulated path as a CSV file. |
| `TGP.py`           | Implementation of the **Adaptive Transfer Gaussian Process (ATGP)**. Defines the transfer kernel across source (simulated) and target (real) tasks, trains hyperparameters using `autograd` and BFGS-style optimization, and provides `predict()` for posterior mean and variance. |
| `IGPR.py`          | Incremental Gaussian Process Regression (online GP). Maintains kernel inverses and updates them efficiently as new data arrives. Not required for every experiment in `main.ipynb`, but showcases online GP implementation. |
| `get_dataset.py`   | Toy 1D dataset generator for transfer-learning demos. Provides simple synthetic functions and sampling routines, used together with `TGP.py` and `run.py` to illustrate transfer learning in an interpretable low-dimensional setting. |
| `util.py`          | Numerical and GP-related utilities. Includes routines such as Cholesky-based linear solves and numerically stable Gaussian log-CDF / tail computations used inside GP and ATGP code. |
| `run.py`           | Example script for toy ATGP experiments. Reads configuration from a `.toml` file, builds a toy source/target dataset via `get_dataset.py`, trains a `TGP` model, and generates diagnostic plots. It is complementary to, but not required by, `main.ipynb`. |

**Data**

| File / Folder                      | Description |
|-----------------------------------|-------------|
| `13个变量的targetdata.csv`        | Real macroeconomic **target** dataset with one `date` column and 13 macro variables. Used by all three models (DSGE-VAR, ATGP, MLP transfer). |
| `simulationcsv/`                  | Folder of simulated DSGE paths, e.g. `sim_data1.csv`, `sim_data2.csv`, … Each file is one macro time series generated under a particular parameter draw. Used to construct priors and pretraining datasets. |

**Generated outputs (examples)**

These are produced when running the notebook:

| File / Folder                          | Description |
|----------------------------------------|-------------|
| `dsgevar.csv`                          | Summary table of DSGE-VAR MAE as a function of prior tightness `λ`. |
| `mlp_tl_1226.csv`                      | Summary table of Neural Network transfer-learning performance. |
| `1225_mae_ATGP_13变量.csv`            | Summary table of ATGP MAE. |
| `./MAE图/*.svg`                        | Publication-style figures comparing MAE across models and variables. |
| `./TL预测结果/`                        | Neural network forecasts vs. realized values for different pretraining setups. |
| `./mlp时间序列/`                       | Time-series of per-period absolute errors for the MLP transfer model. |

The exact filenames may vary slightly depending on which cells are run, but the structure above reflects the main outputs.

---

## 2. High-level workflow

The main notebook `main.ipynb` is organized into sections reflecting the following workflow:

1. **Load and preprocess data**
   - Load real macro data from `13个变量的targetdata.csv`.
   - Construct lagged regressors using `get_lag(df, lags, columns)` from `Preprocess_data.py`.
   - Split into design matrices `X` (lags + intercept) and response matrix `Y` (current values).

2. **Construct DSGE-based priors from simulations**
   - Load simulated paths from `simulationcsv/sim_data*.csv`.
   - For each simulated path, build lagged design matrices and compute:
     - `γ_xx ≈ X'X`,
     - `γ_xy ≈ X'Y`.
   - These matrices define the **Bayesian prior** for the DSGE-VAR model (similar to pseudo-sample sufficient statistics).

3. **DSGE-VAR baseline (Bayesian VAR)**
   - For a grid of prior tightness parameters `λ`:
     - Combine prior (`λT γ_xx`, `λT γ_xy`) and real sample statistics (`XX`, `XY`) to obtain posterior mean of VAR coefficients:
       \[
       \Phi = (λT γ_{xx} + XX)^{-1}(λT γ_{xy} + XY).
       \]
     - Use rolling windows to re-estimate coefficients over time and generate one-step-ahead forecasts.
   - Compute MAE and **standardized MAE** (MAE divided by the unconditional standard deviation of each series).
   - Save results to `dsgevar.csv`.

4. **ATGP (Adaptive Transfer Gaussian Process)**
   - Build a **source** dataset from simulated paths (DSGE) and a **target** dataset from real macro data.
   - Instantiate a `TGP` model from `TGP.py` with:
     - Source inputs/outputs: `src_x`, `src_y`.
     - Target inputs/outputs: `tag_x`, `tag_y`.
   - Train the ATGP to learn:
     - GP hyperparameters (length-scales, variances),
     - Transfer strength parameters that control how much information flows from source to target.
   - Use `predict()` to obtain forecasts for the target time series, compute MAE, and write summary results to e.g. `1225_mae_ATGP_13变量.csv`.

5. **Neural Network transfer learning (MLP)**
   - **Pretraining stage:**
     - For several sample sizes per simulated path (e.g., 10, 20, …, 80), construct large source datasets from DSGE simulations.
     - Train an MLP (defined in `main.ipynb`) to map from lagged macro variables to current values.
     - Save pretrained weights (one model per sample size).
   - **Rolling transfer stage:**
     - For each sample size, load the corresponding pretrained weights.
     - For each rolling window on real data:
       - Fine-tune the MLP on the current window.
       - Forecast the next period.
     - Aggregate predictions, compute standardized MAE, and save the results in `mlp_tl_1226.csv` and per-variable CSVs.

6. **Visualization**
   - Create subplot grids that, for each macro variable and for the aggregate “error,” compare:
     - DSGE-VAR vs. ATGP vs. MLP (MAE as a function of `λ` or an analogous tuning parameter).
   - Save figures as `.svg` files in `./MAE图/`.

---

## 3. How to run this project

### 3.1. Recommended environment

I use Python 3.10+ with the following key packages:

- `numpy`, `pandas`
- `matplotlib`
- `statsmodels`
- `autograd`
- `torch` (PyTorch)
- `toml`

Example setup using `conda`:

```bash
conda create -n econ-ml python=3.11
conda activate econ-ml

pip install numpy pandas matplotlib statsmodels autograd torch toml
```

### 3.2. Data requirements

Make sure the following are available **in the same directory as `main.ipynb`**:

- `13个变量的targetdata.csv`  
  (13-variable macro panel with a `date` column)
- `simulationcsv/`  
  containing the simulated DSGE paths (e.g., `sim_data1.csv`, `sim_data2.csv`, …)

If you want to regenerate simulations from scratch, you can run `Sim_data_gen.py` (or the relevant section in `main.ipynb`), but for the coding sample it is usually sufficient to use the existing CSVs.

### 3.3. Running the main notebook

1. Launch Jupyter or VS Code and open `main.ipynb`.
2. Run the **data preprocessing** cells:
   - Load `13个变量的targetdata.csv`.
   - Construct lagged features via `get_lag`.
3. Run the **DSGE-VAR** section to reproduce the baseline.
4. Run the **ATGP** section to fit and evaluate the transfer GP.
5. Run the **MLP transfer** section to pretrain and fine-tune the neural network.
6. Run the **Visualization** section to generate comparison plots.

Each section is self-contained and can be run independently once the data and preprocessing steps are complete.

---

## 4. Extensibility

The code is intentionally written to be easy to modify:

- **Change lag order or window length**  
  Adjust `lags` and `window_size` (or their corresponding variables) and rerun the relevant section.
- **Extend the λ grid or tuning parameters**  
  Modify the `lambda` array in the DSGE-VAR and ATGP sections or the sample-size grid in MLP pretraining.
- **Experiment with alternative architectures**  
  The MLP architecture and optimizer are defined in a single place in `main.ipynb`; hidden dimensions, layers, and activations can be changed easily.
- **Try different kernels in ATGP**  
  The `TGP` class is modular; custom kernels or transfer structures can be coded there.

---

## 5. Contact

If you have any questions or would like to see additional experiments, extensions, or implementation details, I am happy to discuss them during the interview or via email.
