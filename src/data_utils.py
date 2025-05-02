from data_constants import *
import numpy as np
import pandas as pd


def find_duplicate_columns_by_content(df):
    duplicates = {}
    columns = df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if df[columns[i]].equals(df[columns[j]]):
                duplicates.setdefault(columns[i], []).append(columns[j])
    return duplicates


def load_data_for_tabPFN(file_path):
    """
    Load the data from a CSV file and preps for input to TabPFN.
    Parses the date columns and sets the data types for each column.
    Cuts any data without target values.
    Drops ID columns that should not be inputed in data.
    Returns the data as a pandas DataFrame and the patient IDs by index in new df.
    """
    data = pd.read_csv(file_path, dtype=dtypes, parse_dates=["EXAMDATE", "EXAMDATE_bl"])

    for col in data.select_dtypes(include="datetime"):
        data[col] = data[col].astype(str)

    # drop rows that do not have target values
    for t in targets:
        data = data[data[t].notna()]
    print(f"Data after dropping rows with NA target values, {data.shape} matrix.")
    print(f"Unique patient IDs in the data: {data['PTID'].nunique()}")
    data = data.reset_index(drop=True)
    ids = pd.DataFrame(data["PTID"], columns=["PTID"]).reset_index()

    data = data.drop(columns=id_columns)

    return data, ids
