from __future__ import annotations
import math
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    "gross_margin",
    "operating_margin",
    "net_margin",
    "return_on_assets",
    "liabilities_assets",
    "current_ratio",
    "cash_assets",
    "long_debt_assets",
    "operating_cash_assets",
    "asset_turnover",
    "revenue_growth",
    "eps_growth",
)


def _dict(record):
    return record.to_dict() if hasattr(record, "to_dict") else record


def ratio_features(record):
    values = _dict(record)["values"]

    def val(item, year=-1):
        try:
            return float(values[item][year])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    def div(a, b):
        return a / b if math.isfinite(a) and math.isfinite(b) and b != 0 else float("nan")

    revenue, assets = val("revenue"), val("assets")
    result = [
        div(val("gross_profit"), revenue),
        div(val("operating_income"), revenue),
        div(val("net_income"), revenue),
        div(val("net_income"), assets),
        div(val("liabilities"), assets),
        div(val("current_assets"), val("current_liabilities")),
        div(val("cash"), assets),
        div(val("long_term_debt"), assets),
        div(val("operating_cash_flow"), assets),
        div(revenue, assets),
        div(revenue - val("revenue", 0), abs(val("revenue", 0))),
        div(val("eps_basic") - val("eps_basic", 0), abs(val("eps_basic", 0))),
    ]
    return dict(zip(FEATURE_NAMES, result))


def ratio_vector(record):
    return np.array(list(ratio_features(record).values()), dtype=float)


def _pipeline(C=1.0):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, random_state=2026),
    )


def walk_forward_predictions(records, features, minimum_training=30):
    """Fit yearly models using only earlier outcomes public before the test year's first input.

    Return observation-level probabilities and fold dates so that other forecasters
    can be evaluated on exactly the same test set. Firms may recur across years.
    """
    predictions = []
    for year in sorted({r["outcome_year"] for r in records}):
        test = [r for r in records if r["outcome_year"] == year]
        boundary = min(r["input_filed"] for r in test)
        train = [r for r in records if r["outcome_year"] < year and r["outcome_filed"] < boundary]
        if len(train) < minimum_training or len({r["label"] for r in train}) < 2:
            continue
        model = _pipeline()
        model.fit(np.array([features(r) for r in train], float), [r["label"] for r in train])
        probabilities = model.predict_proba(np.array([features(r) for r in test], float))[:, 1]
        for record, probability in zip(test, probabilities):
            predictions.append(
                {
                    "id": record["id"],
                    "outcome_year": year,
                    "label": record["label"],
                    "probability": float(probability),
                    "training_n": len(train),
                    "latest_training_outcome_filed": max(r["outcome_filed"] for r in train),
                    "test_input_boundary": boundary,
                }
            )
    return predictions
