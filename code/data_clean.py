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
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the cleaned data.")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--vitals_path",
        type=str,
        required=False,
        help="Path to the vitals data file if you want to merge with ADNI db.",
    )
    parser.add_argument(
        "--design_features",
        type=bool,
        default=False,
        help="Whether to include design features in the output.",
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

    # sort by date
    raw_data = raw_data.sort_values(by=["PTID", "EXAMDATE"])

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
    merged_data["ABETA"] = merged_data["ABETA"].replace(f"[<>]", "", regex=True).astype(float)
    merged_data["TAU"] = merged_data["TAU"].replace(f"[<>]", "", regex=True).astype(float)   
    merged_data["ABETA_bl"] = merged_data["ABETA"].replace(f"[<>]", "", regex=True).astype(float)
    merged_data["TAU_bl"] = merged_data["TAU"].replace(f"[<>]", "", regex=True).astype(float)


    # resolving NA values in DX
    with pd.option_context('future.no_silent_downcasting', True):
        merged_data["DX"] = merged_data.groupby("PTID")["DX"].transform(lambda x: x.ffill())
    merged_data["DX"] = merged_data["DX"].fillna("")


    # split the data into training and testing sets by unique PTID
    ids_dx = merged_data[["PTID", "DX"]]
    ids_split = ids_dx.drop_duplicates(subset='PTID', keep='last')
    train_ids, test_ids = train_test_split(
        ids_split,
        test_size=args.test_size,
        random_state=SEED,
        stratify=ids_split["DX"],
    )
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=args.test_size,
        random_state=SEED,
        stratify=train_ids["DX"],
    )
    
    
    # split the data into training, validation, and testing sets
    train_data = merged_data[merged_data["PTID"].isin(train_ids["PTID"])]
    val_data = merged_data[merged_data["PTID"].isin(val_ids["PTID"])]
    test_data = merged_data[merged_data["PTID"].isin(test_ids["PTID"])]
    print(f"Data split into training, validation, and testing sets, {train_data.shape}, {val_data.shape}, {test_data.shape} matrices.")

    # the only transformations we want to do is if any of our fields are not approximately normally distributed. 
    # we kind of should take a look at the distributions of the data before we decide on the transformations.
    # print("Types of each column in the data:")
    # print(merged_data.dtypes)
    # for col in merged_data.columns:
    #     if merged_data[col].dtype == "float64":
    #         plt.hist(merged_data[col], bins=50)
    #         plt.title(f"Distribution of {col}")
    #         plt.xlabel(col)
    #         plt.ylabel("Frequency")
    #         plt.show()

    
    # save the cleaned data
    if args.output_path is not None:
        prefix = "vit_" * args.vitals_path + "transform_" * args.design_features
        os.makedirs(args.output_path, exist_ok=True)
        train_data.to_csv(os.path.join(args.output_path, prefix + "train_data.csv"), index=False)
        val_data.to_csv(os.path.join(args.output_path, prefix + "val_data.csv"), index=False)
        test_data.to_csv(os.path.join(args.output_path, prefix + "test_data.csv"), index=False)
        print(f"Data saved successfully to {args.output_path}.")
