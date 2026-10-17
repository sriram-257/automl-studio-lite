
import json

import joblib
import pytest
from sklearn.datasets import load_breast_cancer

from src.automl import train_and_compare


def test_classification_training_saves_model_and_metadata(tmp_path):
    dataset = load_breast_cancer(as_frame=True)

    result = train_and_compare(
        data=dataset.frame,
        target="target",
        task_type="classification",
        artifacts_dir=str(tmp_path),
    )

    # All three candidate models should be evaluated.
    assert len(result["leaderboard"]) == 3
    assert result["best_model"] in {
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
    }

    # The model and its metadata must be saved.
    model_path = tmp_path / "best_model.joblib"
    metadata_path = tmp_path / "model_metadata.json"

    assert model_path.is_file()
    assert metadata_path.is_file()

    # The saved model must support predictions.
    saved_model = joblib.load(model_path)
    predictions = saved_model.predict(dataset.data.iloc[:3])
    assert len(predictions) == 3

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_name"] == result["best_model"]
    assert metadata["task_type"] == "classification"


def test_missing_target_column_raises_error():
    dataset = load_breast_cancer(as_frame=True)

    with pytest.raises(ValueError, match="Target column"):
        train_and_compare(
            data=dataset.frame,
            target="column_that_does_not_exist",
            task_type="classification",
        )


def test_unsupported_task_type_raises_error():
    dataset = load_breast_cancer(as_frame=True)

    with pytest.raises(ValueError, match="Choose classification or regression"):
        train_and_compare(
            data=dataset.frame,
            target="target",
            task_type="clustering",
        )
