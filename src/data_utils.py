import re
import warnings
from typing import Literal
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from tabpfn_extensions.embedding import TabPFNEmbedding
from tabpfn_extensions import TabPFNRegressor
from tabpfn.constants import (
    NA_PLACEHOLDER,
)

from tabpfn.utils import validate_X_predict, _fix_dtypes

from data_constants import *

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
    # it turns out this was just a problem with PIB and the preprocessing which assumed this was a categorical var
    # data cleaning could have been better

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        embeds = model.model.get_embeddings(X, data_source="test")

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
                e =  _get_embed_wrapper(
                        model, X_train_fold, X_val_fold, data_source=data_source
                    )
                embeddings.append(e)
            return np.concatenate(embeddings, axis=1)
    else:
        raise ValueError("n_fold must be greater than 1.")


# ----- transformer functions -----
    
def reverse_column_transformer_manual(X_transformed, ct, original_feature_count):
    """
    Reverse a ColumnTransformer with a reversible 'encoder' and identity 'remainder'.
    
    Assumes:
        - X_transformed is output from ct.transform(X)
        - ct.transformers_ includes actual fitted transformers and column indices
    """
    X_reconstructed = np.empty((X_transformed.shape[0], original_feature_count), dtype=object)

    current_col = 0

    for name, transformer, cols in ct.transformers_:
        if transformer == 'drop':
            continue
        
        if name == 'remainder':
            # identity passthrough — just copy values
            n_cols = len(cols)
            X_reconstructed[:, cols] = X_transformed[:, current_col:current_col + n_cols]
            current_col += n_cols
            continue

        # reverse-transform encoder
        if hasattr(transformer, 'inverse_transform'):
            n_cols = len(cols)
            transformed_block = X_transformed[:, current_col:current_col + n_cols]
            try:
                reversed_block = transformer.inverse_transform(transformed_block)
                X_reconstructed[:, cols] = reversed_block
            except Exception as e:
                raise RuntimeError(f"Could not reverse-transform '{name}': {e}")
            current_col += n_cols
        else:
            raise RuntimeError(f"Transformer '{name}' does not support inverse_transform")

    return X_reconstructed

def safe_inverse_transform(step, X):
    """Recursively inverse-transform a step, skipping SimpleImputer failures."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    if isinstance(step, SimpleImputer):
        return X  # skip
    elif isinstance(step, Pipeline):
        for sub_name, sub_step in reversed(step.steps):
            if isinstance(sub_step, SimpleImputer):
                continue
            if hasattr(sub_step, 'inverse_transform'):
                try:
                    X = sub_step.inverse_transform(X)
                except Exception:
                    pass
        return X
    elif hasattr(step, 'inverse_transform'):
        try:
            return step.inverse_transform(X)
        except Exception:
            return X
    else:
        return X

def reverse_scale_transformer(ct, X_transformed):
    """
    Partially reverse a ColumnTransformer:
    - Fully reverse 'cats' passthrough block.
    - Reverse 'feat_transform' block, skipping SimpleImputer steps.
    """
    all_cols = [col for _, _, cols in ct.transformers if cols is not None for col in cols]
    max_col = len(all_cols)
    print(max_col)
    n_rows, n_cols = X_transformed.shape
    full_recon = np.empty((n_rows, max_col), dtype=np.float64)

    for name, _, cols in ct.transformers_:
        if name == "cats":
            cats_columns = cols
        elif name == 'feat_transform':
            feat_columns = cols

    # 1. Passthrough block (first 10 columns)
    cat_block = X_transformed[:, cats_columns]
    full_recon[:, cats_columns] = cat_block

    # 2. Feat_transform block
    feat_block = X_transformed[:, feat_columns]
    pipeline: Pipeline = ct.named_transformers_['feat_transform']

    # Manually reverse-transform each step except imputers
    X_step = feat_block
    for name, step in reversed(pipeline.steps):
        X_step = safe_inverse_transform(step, X_step)

    full_recon[:, feat_columns] = X_step
    return full_recon



def reverse_transform(column_transformer, X_transformed, original_input_shape):
    """
    Manually reverses a ColumnTransformer with one-hot and passthrough parts.
    """
    # Get transformers
    onehot_step = column_transformer.named_transformers_['one_hot_encoder']
    onehot_indices = column_transformer.transformers_[0][2]  # [0, 1, ..., 9]

    # Inverse transform the one-hot encoded part
    onehot_end = onehot_step.transform(np.zeros((1, len(onehot_indices)))).shape[1]
    onehot_part = X_transformed[:, :onehot_end]
    onehot_inverse = onehot_step.inverse_transform(onehot_part)

    # Get passthrough part
    n_total = X_transformed.shape[1]
    passthrough_part = X_transformed[:, onehot_end:]

    # Combine both parts
    full = np.empty((X_transformed.shape[0], original_input_shape[1]), dtype=object)
    full[:, onehot_indices] = onehot_inverse

    # Determine passthrough indices
    passthrough_indices = [
        i for i in range(original_input_shape[1]) if i not in onehot_indices
    ]
    full[:, passthrough_indices] = passthrough_part

    return full


def input_processing(
        model,
        X: np.ndarray,
        mode :  Literal[
            "refit",
            "transform",
        ] = "transform",
    ):
    # given a model and inputs, this will return transformed inputs and pipeline.

    # this first section sets up the categoricals and continuous vars.
    # note that the categorical features will move to the front.
    X = X.copy()
    X = validate_X_predict(X, model)
    X = _fix_dtypes(X, cat_indices=model.categorical_features_indices)
    X = model.preprocessor_.transform(X)

    shuffle_idx = [idx for i in [col for _, _, col in model.preprocessor_.transformers_] for idx in i]
    categorical_features = [i for i in range(len(shuffle_idx)) if shuffle_idx[i] in model.categorical_features_indices]
    
    # this second section runs the default tabPFN regression one-hot encoding and normalization schemas.
    # I will spawn models here with only one worker to be deterministic, but this could will work with >1.
    executor = model.executor_
    for i, preprocessor in enumerate(executor.preprocessors):
        if mode == "transform":
            # transform to get input
            X = preprocessor.transform(X).X
        elif mode == "refit":
            X, categorical_features = preprocessor.fit_transform(X, categorical_features)
    
    if len(executor.preprocessors) != 1:
        warnings.warn("Multiple ensembles initialized in model. Only grabbing first pipeline.", UserWarning)

    pipeline = executor.preprocessors[0]
    print(model.preprocessor_)

    return X, pipeline

def remake_columntransformer(ct, new_indices):
    new_transformers = []
    for name, transformer, _ in ct.transformers:
        updated_cols = new_indices[name]
        if updated_cols is not None:
            # Clone the transformer if needed
            if isinstance(transformer, str):
                new_transformers.append((name, transformer, updated_cols))
            else:
                new_transformers.append((name, clone(transformer), updated_cols))

    ct_new = ColumnTransformer(new_transformers, remainder='drop')
    return ct_new

