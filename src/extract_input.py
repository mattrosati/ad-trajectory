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
            