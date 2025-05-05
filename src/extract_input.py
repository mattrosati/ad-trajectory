import numpy as np

import torch
from typing import Iterator

import pandas as pd
from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
import warnings
from typing import Literal
import joblib

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted


from data_utils import *
from data_constants import targets, categorical_features
from sklearn.pipeline import Pipeline


from tabpfn.config import ModelInterfaceConfig
from tabpfn.preprocessing import EnsembleConfig, default_regressor_preprocessor_configs, fit_preprocessing
from tabpfn.model.preprocessing import EncodeCategoricalFeaturesStep, ReshapeFeatureDistributionsStep

from tabpfn.utils import validate_X_predict, _fix_dtypes, _process_text_na_dataframe, infer_random_state, infer_categorical_features

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE reconstruction targets.")
    parser.add_argument(
        "--load-dir",
        type=str,
        default="data/processed",
        help="Directory to load the data from",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/reconstructed",
        help="Directory to save the reconstruction targets to",
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

    config = ModelInterfaceConfig(
        FEATURE_SHIFT_METHOD = None,
        CLASS_SHIFT_METHOD = None,
        FINGERPRINT_FEATURE = False,
    )
    # load the dataset
    # average ensembles before concat
    for f in os.listdir(args.load_dir):
        if f.endswith(".csv"): # TODO: remove train limit
            data, ids = load_data_for_tabPFN(os.path.join(args.load_dir, f))
            print(f"Loaded data from {f}")

            # tell the model what features are categorical
            categorical_features_ = [data.columns.get_loc(col) for col in categorical_features if col in data.columns]
            print(categorical_features_)

            model = TabPFNRegressor(random_state=s,  
                categorical_features_indices=categorical_features_,
                inference_config=config,
                n_estimators = 1
                )

            # fit the model on a dummy target to extract initialized pipeline and ensembles
            x = data
            y = np.arange(x.shape[0])


            model.fit(x, y)

            # TODO: enforce right categorical vars even in embed extraction
            # hacky way to enforce correct transformations of categorical variables
            pipeline = model.executor_.preprocessors[0]
            new_idx = {"cats": categorical_features_, "feat_transform": [i for i in range(x.shape[1]) if i not in categorical_features_]}
            featscaler = pipeline[1].transformer_
            new_featscaler = remake_columntransformer(featscaler, new_idx)
            pipeline[1].transformer_ = new_featscaler

            print(pipeline[2].categorical_transformer_)

            reconstructions, pipeline = input_processing(model, x, "refit")

            # TODO: verify that one can invert transforms (both sequentials and preliminary)
            reverse = reconstructions.copy()
            for i in pipeline[::-1]:
                if isinstance(i, EncodeCategoricalFeaturesStep):
                    reverse = reverse_transform(i.categorical_transformer_, reverse, x.shape)
                elif isinstance(i, ReshapeFeatureDistributionsStep):
                    reverse = reverse_scale_transformer(i.transformer_, reverse)
                    pass
            reverse = reverse_column_transformer_manual(reverse, model.preprocessor_, data.shape[1])
            print(data)
            reverse = pd.DataFrame(reverse, columns=data.columns, index= data.index)
            print(reverse)

            



            # cannot reshuffle reconstructions back to original column order because of one-hot encoding.
            # save reconstruction
            to_save = pd.DataFrame(reconstructions)
            to_save.insert(0, "PTID", ids["PTID"])

            save_path = os.path.join(args.save_dir, f.replace(".csv", "_reconstruct.csv"))
            os.makedirs(args.save_dir, exist_ok=True)
            to_save.to_csv(save_path, index=False)
            print(f"Saved reconstructions to {save_path}")

            # save transforms using joblib
            save_path = os.path.join(args.save_dir,f.replace(".csv", "_pipeline.joblib"))
            joblib.dump(pipeline, save_path)
            