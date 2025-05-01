import random
import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from data_constants import *

# Set the random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


if __name__ == "__main__":
    # initialize the argument parser
    parser = ArgumentParser(description="Data cleaning and preprocessing script.")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file."
    )
    # parser.add_argument("--output_path", type=str, required=True, help="Path to save the cleaned data.")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--vitals_path",
        type=str,
        required=False,
        help="Path to the vitals data file if you want to merge with ADNI db.",
    )

    # parse the arguments
    args = parser.parse_args()

    # load the data
    raw_data = pd.read_csv(
        args.data_path,
        dtype=dtypes,
        parse_dates=["EXAMDATE", "EXAMDATE_bl", "update_stamp"],
        low_memory=False,
    )
    print(f"Data loaded successfully, {raw_data.shape} matrix.")

    # drop columns that are not needed
    raw_data = raw_data.drop(columns=raw_data_drop_columns)
    print(f"Data after dropping columns, {raw_data.shape} matrix.")

    # load vitals data if provided
    if args.vitals_path:
        vitals_data = pd.read_csv(
            args.vitals_path,
            dtype=dtypes,
            parse_dates=["VISDATE", "USERDATE", "USERDATE2", "update_stamp"],
            low_memory=False,
        )

        print(f"Vitals data loaded successfully, {vitals_data.shape} matrix.")
        # drop columns that are not needed
        vitals_data = vitals_data.drop(columns=vitals_drop_columns)
        print(f"Vitals data after dropping columns, {vitals_data.shape} matrix.")

        # merge with the main data based on (
        # raw_data = pd.merge(raw_data, vitals_data, on="IMAGEUID", how="left")
        # print(f"Data merged successfully, {raw_data.shape} matrix.")

    # convert units to SI units

    # replace problematic beta/tau values with maximum values and change to float

    # replace categorical data
    # probably will use dummy encoding for categorical data, but check what tabPFN does

    # split the data into training and testing sets

    # normalize the data based on the training set

    # save the cleaned data
