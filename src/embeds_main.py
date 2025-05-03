import pandas as pd
from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Found unknown categories*"
)


# from tabpfn.config import ModelInterfaceConfig # uncomment this if config necessary

from data_utils import load_data_for_tabPFN, get_embeddings
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

    # config = ModelInterfaceConfig(
    #     FEATURE_SHIFT_METHOD = None,
    #     CLASS_SHIFT_METHOD = None
    # ) # comment this out
    # categorical_features = [0,2,4,6,7,8]

    # load the dataset
    # average ensembles before concat
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
                    clf = TabPFNClassifier(random_state=s, 
                        # inference_config=config, 
                        # categorical_features_indices=categorical_features
                        )
                    clf.feature_names_in_ = x.columns
                    embedding = TabPFNEmbedding(tabpfn_clf=clf, n_fold=args.kfolds)
                else:
                    reg = TabPFNRegressor(random_state=s, 
                        # inference_config=config, 
                        # categorical_features_indices=categorical_features
                        )
                    reg.feature_names_in_ = x.columns
                    embedding = TabPFNEmbedding(tabpfn_reg=reg, n_fold=args.kfolds)

                # fit the model on the data with kfolds if it is training data, with the whole train set if it is val/test data
                if "train" in f:
                    v = get_embeddings(
                        embedding,
                        X_train=x,
                        y_train=y[t],
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
                    v = get_embeddings(
                        embedding,
                        X_train=train_x,
                        y_train=train_y[t],
                        X=x,
                        data_source="test",
                    )  # shape is n_estimators, n_samples, n_features
                v_averaged = np.mean(v, axis=0)
                vecs.append(v_averaged)
                print(f"Shape of the embeddings for {t}: {v.shape}")
            vecs = np.concatenate(vecs, axis=1)
            print(f"Shape of the embeddings: {vecs.shape}")
            to_save = pd.DataFrame(vecs)
            to_save.insert(0, "PTID", ids["PTID"])

            # save the embeddings to a CSV file
            save_path = os.path.join(args.save_dir, f.replace(".csv", "_embeddings.csv"))
            os.makedirs(args.save_dir, exist_ok=True)
            to_save.to_csv(save_path, index=False)
            print(f"Saved embeddings to {save_path}")
