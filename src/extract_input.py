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

from data_utils import load_data_for_tabPFN
from data_constants import targets

from tabpfn.config import ModelInterfaceConfig
from tabpfn.preprocessing import EnsembleConfig

from tabpfn.utils import validate_X_predict, _fix_dtypes, _process_text_na_dataframe

def custom_iter_outputs(
        executor,
        X: np.ndarray,
        *,
        device: torch.device,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        for preprocessor, X_train, y_train, config, cat_ix in zip(
            executor.preprocessors,
            executor.X_trains,
            executor.y_trains,
            executor.ensemble_configs,
            executor.cat_ixs,
        ):
            X_test = preprocessor.transform(X).X
            X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
            # X_test = X_test.unsqueeze(1)

            yield X_test, config


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
        default="data/embeddings",
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
        if f.endswith(".csv") and "train" in str(f):
            data, ids = load_data_for_tabPFN(os.path.join(args.load_dir, f))
            print(f"Loaded data from {f}")

            model = TabPFNRegressor(random_state=s,  
                categorical_features_indices=[0,2,4,6,7,8],
                inference_config=config,
                n_estimators = 1
                )

            # need to fit the model because for some reason I cannot access preprocessing without it.
            y = data[targets]
            x = data.drop(columns=targets)
            vecs = []
            print(x.columns)
            t = targets[0]
            model.fit(x, y[t])

            X = validate_X_predict(x, model)
            X = _fix_dtypes(X, cat_indices=model.categorical_features_indices)

            X = _process_text_na_dataframe(X, ord_encoder=model.preprocessor_)
            final = pd.DataFrame(model.executor_.preprocessors[0].transform(X).X)
            print(model.executor_.preprocessors[0])
            print(pd.DataFrame(X).iloc[:5, :15])
            print(pd.DataFrame(final).iloc[:10, :15])
            print(X.shape)
            print(final.shape)
            break

            # for output, config in custom_iter_outputs(
            #     reg.executor_,
            #     X,
            #     device=reg.device_,
            #     autocast=reg.use_autocast_,
            # ):
            #     print(output, config)
            #     print(output.shape)

            # save



    