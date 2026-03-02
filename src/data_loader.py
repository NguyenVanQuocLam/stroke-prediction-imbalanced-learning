import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
