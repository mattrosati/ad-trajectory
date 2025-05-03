import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
import joblib
from tabpfn.constants import (
    NA_PLACEHOLDER,
)


from data_constants import *
from data_utils import find_duplicate_columns_by_content


def f_to_c(temp_f):
    """
    Convert Fahrenheit to Celsius.
    """
    return (temp_f - 32) * 5.0 / 9.0


def c_to_f(temp_c):
    """
    Convert Celsius to Fahrenheit.
    """
    return (temp_c * 9.0 / 5.0) + 32


def distribution_transform(data, method="yeo-johnson"):
    """
    Apply the Yeo-Johnson transformation or log to the given data.
    """
    pt = PowerTransformer(method=method)
    transformed_data = pt.fit_transform(data)
    return transformed_data


if __name__ == "__main__":
    # initialize the argument parser
    parser = ArgumentParser(description="Data cleaning and preprocessing script.")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file."
    )
    parser.add_argument(
        "--output_path", type=str, required=False, help="Path to save the cleaned data."
    )
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
        default=None,
        help="Path to the vitals data file if you want to merge with ADNI db.",
    )
    parser.add_argument(
        "--design_features",
        type=str,
        default=None,
        help="How to design features in the output. Can be 'log' or 'yeo-johnson'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    s = args.seed
    np.random.seed(s)
    random.seed(s)

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
        print(
            f"Vitals data after dropping rows with NA units, {vitals_data.shape} matrix."
        )

        # correct mislabeled data
        vitals_data["VSTMPUNT"] = vitals_data.apply(
            lambda x: "1" if x["VSTEMP"] > 50 else x["VSTMPUNT"], axis=1
        )
        vitals_data["VSTMPUNT"] = vitals_data.apply(
            lambda x: "2" if x["VSTEMP"] < 50 else x["VSTMPUNT"], axis=1
        )

        # converting
        vitals_data["VSWEIGHT"] = vitals_data.apply(
            lambda x: (
                x["VSWEIGHT"] * unit_conversion["VSWEIGHT"]
                if x["VSWTUNIT"] == "1"
                else x["VSWEIGHT"]
            ),
            axis=1,
        )
        vitals_data["VSTEMP"] = vitals_data.apply(
            lambda x: (
                f_to_c(x["VSTEMP"])
                if (x["VSTMPUNT"] == "1" and x["VSTEMP"] > 50)
                else x["VSTEMP"]
            ),
            axis=1,
        )

        # drop rows with out of liveable range temperature
        vitals_data = vitals_data[vitals_data["VSTEMP"] > 25]
        vitals_data = vitals_data[vitals_data["VSWEIGHT"] > 10]

        print(
            f"Vitals data after converting units and dropping nonsense, {vitals_data.shape} matrix."
        )

        # merge with the main data based on (PTID, VISDATE)
        merged_data = pd.merge(
            raw_data,
            vitals_data,
            left_on=["PTID", "VISCODE"],
            right_on=["PTID", "VISCODE2"],
            how="left",
            validate="one_to_one",
        )

        # final column drop
        merged_data = merged_data.drop(columns=final_drop_columns)

        print(f"Data merged successfully, {merged_data.shape} matrix.")
    else:
        merged_data = raw_data

    # replace problematic beta/tau values with maximum values and change to float
    merged_data["ABETA"] = (
        merged_data["ABETA"].replace(f"[<>]", "", regex=True).astype(float)
    )
    merged_data["TAU"] = (
        merged_data["TAU"].replace(f"[<>]", "", regex=True).astype(float)
    )
    merged_data["ABETA_bl"] = (
        merged_data["ABETA_bl"].replace(f"[<>]", "", regex=True).astype(float)
    )
    merged_data["TAU_bl"] = (
        merged_data["TAU_bl"].replace(f"[<>]", "", regex=True).astype(float)
    )

    # Check any duplicated columns
    duplicates = find_duplicate_columns_by_content(merged_data)
    for key, vals in duplicates.items():
        print(f"{key} is identical to: {', '.join(vals)}")

    # resolving NA values in DX
    with pd.option_context("future.no_silent_downcasting", True):
        merged_data["DX"] = merged_data.groupby("PTID")["DX"].transform(
            lambda x: x.ffill()
        )
    
    # for all string fields, replace with tabPFN placeholder
    categoricals = merged_data.select_dtypes(include=["string", "object"]).columns
    if len(categoricals) > 0:
        merged_data = merged_data.copy()
        merged_data[categoricals] = merged_data[categoricals].fillna(NA_PLACEHOLDER)

    # split the data into training and testing sets by unique PTID
    ids_dx = merged_data[["PTID", "DX"]]
    ids_split = ids_dx.drop_duplicates(subset="PTID", keep="last")
    train_ids, test_ids = train_test_split(
        ids_split,
        test_size=args.test_size,
        random_state=s,
        stratify=ids_split["DX"],
    )
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=args.test_size,
        random_state=s,
        stratify=train_ids["DX"],
    )

    if args.design_features:
        # switch MOCA and MMSE to right skew
        merged_data["MMSE"] = 30 - merged_data["MMSE"]
        merged_data["MOCA"] = 30 - merged_data["MOCA"]
        merged_data["MMSE_bl"] = 30 - merged_data["MMSE_bl"]
        merged_data["MOCA_bl"] = 30 - merged_data["MOCA_bl"]

    if args.design_features == "log":
        # select the columns to apply the tranformation to
        data_to_transform = merged_data[fields_to_transform]
        # apply log transformation to the features
        merged_data[fields_to_transform] = np.log1p(merged_data[fields_to_transform])

    # split the data into training, validation, and testing sets
    train_data = merged_data[merged_data["PTID"].isin(train_ids["PTID"])].copy()
    val_data = merged_data[merged_data["PTID"].isin(val_ids["PTID"])].copy()
    test_data = merged_data[merged_data["PTID"].isin(test_ids["PTID"])].copy()
    print(
        f"Data split into training, validation, and testing sets, {train_data.shape}, {val_data.shape}, {test_data.shape} matrices."
    )

    if args.design_features == "yeo-johnson":
        # apply Yeo-Johnson transformation to the features
        pt = PowerTransformer(method="yeo-johnson")
        tr_train_data = pt.fit_transform(train_data[fields_to_transform])
        tr_val_data = pt.transform(val_data[fields_to_transform])
        tr_test_data = pt.transform(test_data[fields_to_transform])
        train_data[fields_to_transform] = pd.DataFrame(
            tr_train_data, columns=pt.get_feature_names_out(), index=train_data.index
        )
        val_data[fields_to_transform] = pd.DataFrame(
            tr_val_data, columns=pt.get_feature_names_out(), index=val_data.index
        )
        test_data[fields_to_transform] = pd.DataFrame(
            tr_test_data, columns=pt.get_feature_names_out(), index=test_data.index
        )

    # verify normality by statistical test and visual inspection
    # for col in val_data[fields_to_transform].columns:
    #     stat, p = shapiro(val_data[col], nan_policy='omit')
    #     print(col + ": " + "p-value =", p)

    # for col in val_data.columns:
    #     if val_data[col].dtype == "float64":
    #         plt.hist(val_data[col], bins=20)
    #         plt.title(f"Distribution of {col}")
    #         plt.xlabel(col)
    #         plt.ylabel("Frequency")
    #         plt.show()

    print("Columns in train data: ", train_data.columns)

    # save the cleaned data
    if args.output_path is not None:
        design_feature_string = (
            args.design_features + "_" if args.design_features else ""
        )
        vitals_string = "vitals_" if args.vitals_path else ""
        prefix = vitals_string + design_feature_string
        os.makedirs(args.output_path, exist_ok=True)
        train_data.to_csv(
            os.path.join(args.output_path, prefix + "train_data.csv"), index=False
        )
        val_data.to_csv(
            os.path.join(args.output_path, prefix + "val_data.csv"), index=False
        )
        test_data.to_csv(
            os.path.join(args.output_path, prefix + "test_data.csv"), index=False
        )
        print(f"Data saved successfully to {args.output_path}.")

        if args.design_features == "yeo-johnson":
            # save the transformer
            joblib.dump(
                pt, os.path.join(args.output_path, "yeo-johnson_transformer.pkl")
            )
            print(f"Transformer saved successfully to {args.output_path}.")
