import pandas as pd
from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
import warnings

warnings.simplefilter("error")

from data_utils import load_data_for_tabPFN
from data_constants import targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN Embedding")
    parser.add_argument(
        "--kfolds",
        type=int,
        default=10,
        help="Number of folds for cross-validation for building robust embeddings",
    )
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
        help="Directory to save the embeddings to",
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

    # load the dataset
    # set the number of ensembles to 8 for both regression and classification
    for f in os.listdir(args.load_dir):
        if f.endswith(".csv"):
            data, ids = load_data_for_tabPFN(os.path.join(args.load_dir, f))
            print(f"Loaded data from {f}")

            # fit the model and extract the embeddings
            y = data[targets]
            x = data.drop(columns=targets)
            vecs = []
            for t in tqdm(targets):
                # TODO: store the preprocessed data for input to TabPFN
                # most likely will require a couple of lines of the sort:
                # from tabpfn.utils import ModelInterfaceConfig
                # from tabpfn import TabPFNClassifier
                # config = ModelInterfaceConfig(
                #     shift_labels=False,       # ❌ don't shift class labels
                #     permute_features=False,   # ❌ don't shuffle feature columns
                #     normalize_by_median=True  # ✅ (optional) still normalize numerics
                # )
                # model = TabPFNClassifier(interface_config=config)
                # model.fit(X_train, y_train)
                # x = validate_X_predict(x, reg)
                # x = _fix_dtypes(x, cat_indices=reg.categorical_features_indices)
                # tran = reg.preprocessor_.transform(x)

                # make model for each target
                # if the target is a classification task, use TabPFNClassifier
                # if the target is a regression task, use TabPFNRegressor
                if t == "DX":
                    clf = TabPFNClassifier(random_state=s)
                    clf.feature_names_in_ = x.columns
                    embedding = TabPFNEmbedding(tabpfn_clf=clf, n_fold=args.kfolds)
                else:
                    reg = TabPFNRegressor(random_state=s)
                    reg.feature_names_in_ = x.columns
                    embedding = TabPFNEmbedding(tabpfn_reg=reg, n_fold=args.kfolds)

                # fit the model on the data with kfolds if it is training data, with the whole train set if it is val/test data
                if "train" in f:
                    v = embedding.get_embeddings(
                        X_train=x.to_numpy(),
                        y_train=y[t].to_numpy(),
                        X=None,
                        data_source="train",
                    )
                else:
                    print("Fitting whole train set to get embeddings for val/test set.")

                    # find the corresponding train set
                    train_set = f.replace("val", "train").replace("test", "train")
                    train_data, _ = load_data_for_tabPFN(
                        os.path.join(args.load_dir, train_set)
                    )
                    train_x = train_data.drop(columns=targets)
                    train_y = train_data[targets]
                    v = embedding.get_embeddings(
                        X_train=train_x.to_numpy(),
                        y_train=train_y[t].to_numpy(),
                        X=x.to_numpy(),
                        data_source="test",
                    ) # shape is n_estimators, n_samples, n_features
                v_averaged = np.mean(v, axis=0)
                vecs.append(v_averaged)
                print(f"Shape of the embeddings for {t}: {v.shape}")
            vecs = np.concatenate(vecs, axis=1)
            print(f"Shape of the embeddings: {vecs.shape}")
            to_save = np.concatenate((ids["PTID"].to_numpy(), vecs), axis=1)

        # save the embeddings to a CSV file
        save_path = os.path.join(args.save_dir, f.replace(".csv", "_embeddings.csv"))
