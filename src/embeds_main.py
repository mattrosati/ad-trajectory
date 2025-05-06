import pandas as pd
from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding
from data_constants import categorical_features, targets

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
import re

import joblib

from datasets import Dataset, DatasetDict

from sklearn.preprocessing import StandardScaler, LabelEncoder


# from tabpfn.config import ModelInterfaceConfig # uncomment this if config necessary

from data_utils import load_data_for_tabPFN, get_embeddings


def scale_targets(y, cat_names):
    encoder_dict = {}
    for i in y.columns:
        if i in cat_names:
            encoder = LabelEncoder()
            y[i] = encoder.fit_transform(y[i])
        else:
            encoder = StandardScaler()
            y[i] = encoder.fit_transform(y[i].to_frame())
        encoder_dict[i] = encoder

    return y, encoder_dict


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
    # average ensembles before concat
    dataset_dict = {}
    transforms_dict = {}
    for f in os.listdir(args.load_dir):
        if f.endswith(".csv") and "vitals" not in f:
            data, ids = load_data_for_tabPFN(os.path.join(args.load_dir, f))
            print(f"Loaded data from {f}")

            # fit the model and extract the embeddings
            y = data[targets]
            x = data.drop(columns=targets)

            # store time!!!
            time_vals = x["Month_bl"].rename("months")

            # normalize targets
            transformed_y = None
            if "train" in f:
                transformed_y, transforms_f = scale_targets(y.copy(), categorical_features)

            # tell the model what features are categorical
            categorical_features_ = [x.columns.get_loc(col) for col in categorical_features if col in x.columns]

            vecs = []
            for t in tqdm(targets):
                # make model for each target
                # if the target is a classification task, use TabPFNClassifier
                # if the target is a regression task, use TabPFNRegressor
                if t == "DX":
                    clf = TabPFNClassifier(random_state=s, 
                        # inference_config=config, 
                        categorical_features_indices=categorical_features_
                        )
                    clf.feature_names_in_ = x.columns
                    embedding = TabPFNEmbedding(tabpfn_clf=clf, n_fold=args.kfolds)
                else:
                    reg = TabPFNRegressor(random_state=s, 
                        # inference_config=config, 
                        categorical_features_indices=categorical_features_
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
            idx = ids["PTID"].reset_index()


            # make big pd.df with embeddings, labels, PTIDs and idx
            feature_cols = [f"f_{i}" for i in range(vecs.shape[1])]
            to_save = pd.DataFrame(vecs, columns=feature_cols)
            if transformed_y is not None:
                to_save[transformed_y.columns] = transformed_y.reset_index(drop=True)
            else:
                to_save[y.columns] = y.reset_index(drop=True)
            to_save[idx.columns] = idx
            to_save[time_vals.name] = time_vals.reset_index(drop=True)

            # figure out what split this is
            for split in ["train", "test", "val"]:
                if split in f.lower():
                    split_add = split
            # figure out what data this is
            preproc_key = re.sub(r"_?(train|test|val)_data\.csv", "", f)
            if preproc_key == "":
                preproc_key = "base"

            if preproc_key not in dataset_dict:
                dataset_dict[preproc_key] = {}
            if preproc_key not in transforms_dict:
                transforms_dict[preproc_key] = {}

            dataset_dict[preproc_key][split_add] = to_save
            if transformed_y is not None:
                transforms_dict[preproc_key] = transforms_f


    # traverse the whole datasets dict to: 
    # 1. transform test and val based on train transform
    # 2. convert to huggingface dataset
    print("saving and transforming test/val")
    for pre_key in dataset_dict.keys():
        for split in ["train", "test", "val"]:
            df = dataset_dict[pre_key][split]
            if split == "test" or split == "val":
                for col in targets:
                    to_transform = df[col]
                    t = transforms_dict[preproc_key][col]
                    print(preproc_key, split, col)
                    print(t)
                    if col in categorical_features:
                        df[col] = t.transform(to_transform)
                        print("Categories:", t.classes_)  # list of arrays
                    else:
                        df[col] = t.transform(to_transform.to_frame())

            # Create Hugging Face Dataset
            dataset_dict[pre_key][split] = Dataset.from_pandas(df)

    # for each dataset dict in the whole set, save
    for pre_key in dataset_dict.keys():
        directory = os.path.join(args.save_dir, pre_key)
        dataset_object = DatasetDict(dataset_dict[pre_key])
        dataset_object.save_to_disk(directory)
        label_transformers = transforms_dict[preproc_key]
        joblib.dump(label_transformers, os.path.join(directory, "label_transformers.pkl"))

    print("saved all!")

