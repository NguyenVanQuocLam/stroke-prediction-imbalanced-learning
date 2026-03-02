import pandas as pd
from imblearn.over_sampling import SMOTENC
from ctgan import CTGAN


def apply_smote(X_train, y_train):

    categorical_features = X_train.select_dtypes(include=["object"]).columns
    categorical_indices = [X_train.columns.get_loc(col) for col in categorical_features]

    smote_nc = SMOTENC(
        categorical_features=categorical_indices,
        random_state=42
    )

    X_res, y_res = smote_nc.fit_resample(X_train, y_train)

    return X_res, y_res


def apply_ctgan(X_train, y_train, epochs=300):

    train_df = pd.concat([X_train, y_train], axis=1)

    minority = train_df[train_df["stroke"] == 1]

    categorical_cols = minority.select_dtypes(include=["object"]).columns.tolist()

    ctgan = CTGAN(
        epochs=epochs,
        verbose=True
    )

    ctgan.fit(minority, categorical_cols)

    n_samples = len(train_df[train_df["stroke"] == 0]) - len(minority)

    synthetic = ctgan.sample(n_samples)
    synthetic["stroke"] = 1

    balanced_train = pd.concat([train_df, synthetic], ignore_index=True)

    X_gan = balanced_train.drop("stroke", axis=1)
    y_gan = balanced_train["stroke"]

    return X_gan, y_gan
