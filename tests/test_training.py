from pathlib import Path
import subprocess


def test_training_creates_model():
    subprocess.run(
        ["python", "scripts/train.py"],
        check=True,
    )

    model_path = Path("artifacts/models/baseline_model.pkl")

    assert model_path.exists()
