import random
import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

dtypes = {
    "RID": "string",
    "COLPROT": "string",
    "ORIGPROT": "string", 
    "PTID": "string", 
    "VISCODE": "string",
    "DX_BL": "string",
    "PTGENDER": "string",
    "PTEDUCAT": "float64",
    "PTETHCAT": "string",
    "PTRACCAT": "string",
    "PTMARRY": "string",
    "ABETA": "string", #'>1700' giving problems
    "TAU": "string", #'>1300' giving problems
    "PTAU": "string", #'>120' giving problems
    "FSVERSION": "string",
    "DX": "string",
    "FLDSTRENG_bl": "string",
    "FSVERSION_bl": "string",
    "ABETA_bl": "string", #'>1700' giving problems
    "TAU_bl": "string", #'>1300' giving problems
    "PTAU_bl": "string", #'>120' giving problems
    }


if __name__ == "__main__":
    # initialize the argument parser
    parser = ArgumentParser(description="Data cleaning and preprocessing script.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    # parser.add_argument("--output_path", type=str, required=True, help="Path to save the cleaned data.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")

    # parse the arguments
    args = parser.parse_args()

    # load the data
    raw_data = pd.read_csv(args.data_path, dtype=dtypes, parse_dates=["EXAMDATE", "EXAMDATE_bl", "update_stamp"], low_memory=False)
    print("Data loaded successfully, {raw_data.shape} matrix.")

    # drop columns that are not needed

    # replace problematic beta/tau values with maximum values and change to float
    
    # replace categorical data
    # probably will use dummy encoding for categorical data, but check what tabPFN does

    # split the data into training and testing sets


    # normalize the data based on the training set



    # save the cleaned data