from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_logistic(preprocessor):

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    return model
