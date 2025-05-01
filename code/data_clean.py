import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from data_constants import *

# Set the random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def f_to_c(temp_f):
    """
    Convert Fahrenheit to Celsius.
    """
    return (temp_f - 32) * 5.0/9.0

def c_to_f(temp_c):
    """
    Convert Celsius to Fahrenheit.
    """
    return (temp_c * 9.0/5.0) + 32


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
        # drop columns that are not needed. Removing height data because it is extremely low quality
        vitals_data = vitals_data.drop(columns=vitals_drop_columns)
        print(f"Vitals data after dropping columns, {vitals_data.shape} matrix.")

        # drop vitals data with Na visdate or viscode to merge with the main data
        # TODO: return if we continue this project because there is likely a more complex merge that can be done to keep all data
        vitals_data = vitals_data[vitals_data["VISDATE"].notna()]
        vitals_data = vitals_data[vitals_data["VISCODE2"].notna()]


        # convert units to SI units
        # drop data if units are NA or temp source is NA
        vitals_data = vitals_data[vitals_data["VSTMPSRC"].notna()]
        vitals_data = vitals_data[vitals_data["VSWTUNIT"].notna()]
        vitals_data = vitals_data[vitals_data["VSTMPUNT"].notna()]

        # drop rows with zero temperature
        vitals_data = vitals_data[vitals_data["VSTEMP"] > 20]
        print(f"Vitals data after dropping rows with NA units, {vitals_data.shape} matrix.")

        # correct mislabeled data
        vitals_data["VSTMPUNT"] = vitals_data.apply(lambda x: "1" if x["VSTEMP"] > 50 else x["VSTMPUNT"], axis=1)
        vitals_data["VSTMPUNT"] = vitals_data.apply(lambda x: "2" if x["VSTEMP"] < 50 else x["VSTMPUNT"], axis=1)

        # converting 
        vitals_data["VSWEIGHT"] = vitals_data.apply(lambda x: x["VSWEIGHT"] * unit_conversion["VSWEIGHT"] if x["VSWTUNIT"] == "1" else x["VSWEIGHT"], axis=1)
        vitals_data["VSTEMP"] = vitals_data.apply(lambda x: f_to_c(x["VSTEMP"]) if (x["VSTMPUNT"] == "1" and x["VSTEMP"] > 50) else x["VSTEMP"], axis=1)
        
        # drop rows with out of liveable range temperature
        vitals_data = vitals_data[vitals_data["VSTEMP"] > 25]
        vitals_data = vitals_data[vitals_data["VSWEIGHT"] > 10]

        print(f"Vitals data after converting units and dropping nonsense, {vitals_data.shape} matrix.")

        # merge with the main data based on (PTID, VISDATE)
        merged_data = pd.merge(
            raw_data,
            vitals_data,
            left_on=["PTID", "VISCODE"],
            right_on=["PTID", "VISCODE2"],
            how="left",
            validate="one_to_one",
        )
        print(f"Data merged successfully, {merged_data.shape} matrix.")

    # replace problematic beta/tau values with maximum values and change to float

    # replace categorical data
    # probably will use dummy encoding for categorical data, but check what tabPFN does

    # split the data into training and testing sets

    # normalize the data based on the training set

    # save the cleaned data
