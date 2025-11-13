import pandas as pd
import numpy as np


def extract_feature(sum_df, lag, columns):
    df_processed = sum_df.copy()
    for lag in range(0, lag):
        df_processed[[co+str(lag+1) for co in columns]] =  sum_df[[co+str(lag+1) for co in columns]]
    df_processed = df_processed.dropna()
    return df_processed

def get_lag(ini_df,lag,columns):
    df_processed = ini_df.copy()
    for lag in range(0, lag):
        df_processed[[co+str(lag+1) for co in columns]] =  ini_df[columns].shift(lag+1)
    df_processed = df_processed.dropna()
    return df_processed    

