import typing
import re
import warnings

import torch
import numpy as np
import pandas as pd

from data_constants import *

from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding
from tabpfn.constants import (
    NA_PLACEHOLDER,
)

def extract_split(s):
    match = re.search(r'(test|val|train)', s)
    return match.group(0) if match else None


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
    data = pd.read_csv(file_path, parse_dates=["EXAMDATE", "EXAMDATE_bl"])

    for col in data.select_dtypes(include="object"):
        data[col] = data[col].astype("string")

    # drop rows that do not have target values
    for t in targets:
        data = data[data[t].notna()]
    print(f"Data after dropping rows with NA target values, {data.shape} matrix.")
    print(f"Unique patient IDs in the data: {data['PTID'].nunique()}")
    ids = pd.DataFrame(data["PTID"], columns=["PTID"])

    # convert date columns to floats
    for col in data.select_dtypes(include="datetime"):
        data[col] = (data[col] - pd.Timestamp("1970-01-01")).dt.total_seconds() / (
            60 * 60 * 24
        )  # converting to days since 1970

    data = data.drop(columns=id_columns)

    return data, ids


def _get_embed_wrapper(
    model: TabPFNEmbedding,
    X_train,
    X,
    data_source: str,
):
    # I built this wrapper to figure out why the encoding was giving a warning.
    # It turns out that scikit-learn has added a warn_on_unknown parameter that will raise a warning
    # if handle_unknown="ignore" and drop is not None, which is what the OneHotEncoder in TabPFN is doing.
    # This does not change the behavior, which is what I want. To avoid this warning then I only need to filter it out.
    # Maybe I could propose a change to the TabPFN code.

    # catch and filter out the warning.
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", UserWarning)
    # it turns out this was just a problem with PIB and the preprocessing which assumed this was a categorical var
    # data cleaning could have been better

    embeds = model.model.get_embeddings(X, data_source=data_source)
    return embeds


def get_embeddings(
    model: TabPFNEmbedding,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X: np.ndarray,
    data_source: str,
) -> np.ndarray:
    """Extracts embeddings for the given dataset using the trained model.
    This is my own version because I want to check if the test set has categories the model has never seen before and fix it.

    Args:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target labels.
        X (np.ndarray): Data for which embeddings are to be extracted.
        data_source (str): Specifies the data source ("test" for test data).

    Returns:
        np.ndarray: The extracted embeddings.

    Raises:
        ValueError: If no model is set before calling get_embeddings.

    """
    if model.model is None:
        raise ValueError("No model has been set.")

    # If no cross-validation is used, train and return embeddings directly
    if model.n_fold == 0:
        model.model.fit(X_train, y_train)
        return _get_embed_wrapper(model, X_train, X, data_source=data_source)
    elif model.n_fold >= 2:
        if data_source == "test":
            model.model.fit(X_train, y_train)
            return _get_embed_wrapper(model, X_train, X, data_source=data_source)
        else:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=model.n_fold, shuffle=False)
            embeddings = []
            for train_index, val_index in kf.split(X_train):

                # converted all of this to pandas
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_fold, _y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
                model.model.fit(X_train_fold, y_train_fold)
                embeddings.append(
                    _get_embed_wrapper(
                        model, X_train_fold, X_val_fold, data_source=data_source
                    ),
                )
            return np.concatenate(embeddings, axis=1)
    else:
        raise ValueError("n_fold must be greater than 1.")
