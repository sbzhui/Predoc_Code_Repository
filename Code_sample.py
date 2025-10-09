# -*- coding: utf-8 -*-
"""
Refactored script for performing rolling window forecasting on macroeconomic data
using a Vector Autoregression (VAR) model with priors from simulation data.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm # A library to display progress bars

# --- Configuration & Constants ---

# File paths
TARGET_DATA_PATH = '13Var_targetdata.csv'
SIMULATION_DATA_DIR = './simulationcsv/'
OUTPUT_CSV_PATH = 'dsgevar_results.csv'

# Model parameters
NUM_VARIABLES = 13
LAGS = 3
NUM_SIMULATIONS = 200
WINDOW_SIZE = 80

def get_lag(df, lags, columns):
    """
    Creates lagged features for the specified columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        lags (int): The number of lags to create.
        columns (list): A list of column names to lag.

    Returns:
        pd.DataFrame: The DataFrame with new lagged columns.
    """
    df_lagged = df.copy()
    for lag in range(1, lags + 1):
        for col in columns:
            df_lagged[f'{col}{lag}'] = df_lagged[col].shift(lag)
    return df_lagged

def load_target_data(filepath, lags):
    """
    Loads and preprocesses the real-world target data.

    Args:
        filepath (str): The path to the target data CSV file.
        lags (int): The number of lags to create as features.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The preprocessed DataFrame.
            - list: The names of the target variable columns.
    """
    # Read the data
    df = pd.read_csv(filepath)
    columns = df.columns.drop('date')

    # Create lagged features and set the index
    df = get_lag(df, lags, columns)
    df.set_index('date', inplace=True)
    
    return df, columns

def compute_simulation_priors(sim_dir, num_simulations, columns, lags):
    """
    Computes the prior matrices (gamma_xx and gamma_xy) from simulation data.
    These priors are used to inform the VAR model estimation.

    Args:
        sim_dir (str): The directory containing the simulation CSV files.
        num_simulations (int): The number of simulation files to process.
        columns (list): The names of the target variable columns.
        lags (int): The number of lags used in the model.

    Returns:
        tuple: A tuple containing:
            - list: A list of gamma_xx matrices (X'X for each simulation).
            - list: A list of gamma_xy matrices (X'Y for each simulation).
    """
    print("Computing priors from simulation data...")
    gamma_xx_list = []
    gamma_xy_list = []

    for i in tqdm(range(1, num_simulations + 1)):
        # Load and prepare a single simulation dataset
        prior_df = pd.read_csv(f'{sim_dir}sim_data{i}.csv').iloc[:NUM_VARIABLES, 20:200].T
        prior_df.columns = columns # Ensure columns have the correct names

        # Create lagged features for the simulation data
        prior_x_df = pd.DataFrame(index=prior_df.index)
        for lag in range(1, lags + 1):
            for col in columns:
                prior_x_df[f'{col}{lag}'] = prior_df[col].shift(lag)
        
        prior_x_df = sm.add_constant(prior_x_df, has_constant='add')
        
        # Align Y and X by dropping rows with NaNs from lagging
        combined = pd.concat([prior_df, prior_x_df], axis=1).dropna()
        prior_y = combined[columns].values
        prior_x = combined.drop(columns, axis=1).values
        
        # Calculate and store the X'X and X'Y matrices
        gamma_xx_list.append(prior_x.T @ prior_x)
        gamma_xy_list.append(prior_x.T @ prior_y)
        
    return gamma_xx_list, gamma_xy_list

def perform_rolling_forecast(df, columns, gamma_xx, gamma_xy, lambda_val):
    """
    Performs the main rolling window forecast loop.

    Args:
        df (pd.DataFrame): The preprocessed target data.
        columns (list): The names of the target variable columns.
        gamma_xx (list): The list of prior X'X matrices.
        gamma_xy (list): The list of prior X'Y matrices.
        lambda_val (float): The shrinkage parameter lambda.

    Returns:
        pd.DataFrame: A DataFrame containing the Mean Absolute Error (MAE) for each variable.
    """
    # Prepare the exogenous (X) and endogenous (Y) variables from the real data
    feature_cols = [f'{col}{lag}' for col in columns for lag in range(1, LAGS + 1)]
    X = sm.add_constant(df[feature_cols])
    Y = df[columns]

    predictions_list = []
    
    # Start the rolling window loop
    for i in range(len(df) - WINDOW_SIZE):
        # Define the training window for this iteration
        start_index = i
        end_index = i + WINDOW_SIZE
        
        # Get window data, dropping NaNs which appear at the start
        real_x_window = X.iloc[start_index:end_index].dropna()
        real_y_window = Y.loc[real_x_window.index]

        # Convert to numpy for efficiency
        real_x_np = real_x_window.values
        real_y_np = real_y_window.values
        
        # Calculate X'X and X'Y from the real data window
        XX = real_x_np.T @ real_x_np
        XY = real_x_np.T @ real_y_np

        # Estimate the VAR coefficients (PHI) by combining real data with priors
        phi_sum = np.zeros((1 + LAGS * NUM_VARIABLES, NUM_VARIABLES))
        for p_xx, p_xy in zip(gamma_xx, gamma_xy):
            # Calculate PHI for one simulation prior
            inv_matrix = np.linalg.inv(lambda_val * WINDOW_SIZE * p_xx + XX)
            phi = inv_matrix @ (lambda_val * WINDOW_SIZE * p_xy + XY)
            phi_sum += phi
        
        # Average the coefficients from all simulations
        phi_final = phi_sum / NUM_SIMULATIONS

        # Make a one-step-ahead prediction
        x_predict = X.iloc[end_index].values
        prediction = x_predict @ phi_final
        predictions_list.append(prediction)

    # Convert list of predictions to a DataFrame after the loop
    predicts_df = pd.DataFrame(predictions_list, columns=columns, index=Y.index[WINDOW_SIZE:])
    
    # Calculate and return the Mean Absolute Error (MAE)
    mae = (Y.iloc[WINDOW_SIZE:] - predicts_df).abs().mean().to_frame().T
    return mae

def main():
    """
    Main function to run the entire forecasting and evaluation pipeline.
    """
    # Load and preprocess the target data
    df, columns = load_target_data(TARGET_DATA_PATH, LAGS)

    # Compute the priors from all simulation files
    gamma_xx, gamma_xy = compute_simulation_priors(SIMULATION_DATA_DIR, NUM_SIMULATIONS, columns, LAGS)

    # Define the grid of lambda values for hyperparameter tuning
    lambda_values_1 = np.linspace(0.1, 1, 10)
    lambda_values_2 = np.linspace(2, 10, 9)
    lambda_values = np.concatenate((lambda_values_1, lambda_values_2))

    # Store MAE results for each lambda
    all_mae_results = []

    print(f"\nStarting rolling forecast for {len(lambda_values)} lambda values...")
    for lam in tqdm(lambda_values):
        mae_result = perform_rolling_forecast(df, columns, gamma_xx, gamma_xy, lam)
        mae_result['lambda'] = lam
        all_mae_results.append(mae_result)
        
    # Combine all results into a single DataFrame
    final_results_df = pd.concat(all_mae_results).set_index('lambda')
    
    # Calculate a final scaled error metric
    stdevs = df[columns].iloc[WINDOW_SIZE:].std().values
    scaled_errors = final_results_df[columns] / stdevs
    final_results_df['scaled_error_mean'] = scaled_errors.mean(axis=1)

    # Save the results to a CSV file
    final_results_df.to_csv(OUTPUT_CSV_PATH)
    
    print(f"\nAnalysis complete. Results saved to '{OUTPUT_CSV_PATH}'")
    print("\nFinal MAE and Scaled Error for each Lambda:")
    print(final_results_df)

if __name__ == '__main__':
    main()