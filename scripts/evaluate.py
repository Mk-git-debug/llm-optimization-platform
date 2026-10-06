from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    data = load_breast_cancer()

    _, X_test, _, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model_path = Path("artifacts/models/baseline_model.pkl")
    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")


if __name__ == "__main__":
    main()
