"""Check chronological exclusions and alignment of held-out benchmark predictions."""

import numpy as np

from numeraire import benchmark


def test_walk_forward_excludes_late_labels_and_preserves_test_ids(monkeypatch):
    fits = []

    class Recorder:
        def fit(self, features, labels):
            fits.append((features[:, 0].tolist(), list(labels)))

        def predict_proba(self, features):
            p = features[:, 0] / 10
            return np.column_stack([1 - p, p])

    monkeypatch.setattr(benchmark, "_pipeline", Recorder)
    records = [
        {
            "id": "early-a",
            "outcome_year": 2020,
            "label": 0,
            "input_filed": "2020-02-01",
            "outcome_filed": "2021-02-01",
            "feature": 1,
        },
        {
            "id": "early-b",
            "outcome_year": 2020,
            "label": 1,
            "input_filed": "2020-02-01",
            "outcome_filed": "2021-02-02",
            "feature": 2,
        },
        {
            "id": "unavailable",
            "outcome_year": 2020,
            "label": 1,
            "input_filed": "2020-02-01",
            "outcome_filed": "2021-03-01",
            "feature": 9,
        },
        {
            "id": "test-b",
            "outcome_year": 2021,
            "label": 0,
            "input_filed": "2021-02-10",
            "outcome_filed": "2022-02-01",
            "feature": 4,
        },
        {
            "id": "test-a",
            "outcome_year": 2021,
            "label": 1,
            "input_filed": "2021-02-03",
            "outcome_filed": "2022-02-01",
            "feature": 3,
        },
    ]
    result = benchmark.walk_forward_predictions(
        records, lambda r: [r["feature"]], minimum_training=2
    )
    assert fits == [([1.0, 2.0], [0, 1])]
    assert [(r["id"], r["probability"]) for r in result] == [("test-b", 0.4), ("test-a", 0.3)]
    assert all(r["latest_training_outcome_filed"] < r["test_input_boundary"] for r in result)


def test_walk_forward_does_not_fit_one_class(monkeypatch):
    def unexpected_fit():
        raise AssertionError("A one-class training sample must not be fitted")

    monkeypatch.setattr(benchmark, "_pipeline", unexpected_fit)
    records = [
        {
            "id": "train",
            "outcome_year": 2020,
            "label": 1,
            "input_filed": "2020-02-01",
            "outcome_filed": "2021-02-01",
        },
        {
            "id": "test",
            "outcome_year": 2021,
            "label": 0,
            "input_filed": "2021-02-02",
            "outcome_filed": "2022-02-01",
        },
    ]
    assert benchmark.walk_forward_predictions(records, lambda r: [1], minimum_training=1) == []
