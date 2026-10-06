from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    output_dir = Path("artifacts/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "baseline_model.pkl"
    joblib.dump(model, model_path)

    print("Model trained successfully.")
    print(f"Saved to: {model_path}")


if __name__ == "__main__":
    main()
