from sklearn.metrics import classification_report, roc_auc_score
from data_loader import load_data
from preprocessing import build_preprocessor
from models import build_logistic
from oversampling import apply_smote, apply_ctgan
from fairness import fairness_check


def train_evaluate(X_train, y_train, X_test, y_test, label):

    preprocessor = build_preprocessor(X_train)
    model = build_logistic(preprocessor)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {label} =====")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    if "gender" in X_test.columns:
        print("\nFairness - Gender:")
        print(fairness_check(model, X_test, y_test, "gender"))

    if "Residence_type" in X_test.columns:
        print("\nFairness - Residence_type:")
        print(fairness_check(model, X_test, y_test, "Residence_type"))


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data("stroke_dataset.csv")

    # No balancing
    train_evaluate(X_train, y_train, X_test, y_test, "No Balancing")

    # SMOTE
    X_smote, y_smote = apply_smote(X_train, y_train)
    train_evaluate(X_smote, y_smote, X_test, y_test, "SMOTE-NC")

    # CTGAN
    X_gan, y_gan = apply_ctgan(X_train, y_train)
    train_evaluate(X_gan, y_gan, X_test, y_test, "CTGAN Oversampling")
