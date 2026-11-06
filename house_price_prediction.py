"""Train, evaluate, and demonstrate a synthetic house-price model."""

import argparse
import csv
from pathlib import Path
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "price"
NUMERIC_FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_to_city_km",
    "parking_spaces",
]
CATEGORICAL_FEATURES = ["location", "condition"]
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], list[float]]:
    """Read and validate the CSV data required by the model."""
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        features = []
        targets = []
        for line_number, row in enumerate(reader, start=2):
            try:
                feature = {
                    name: float(row[name]) for name in NUMERIC_FEATURES
                }
                feature.update({name: row[name].strip() for name in CATEGORICAL_FEATURES})
                target = float(row[TARGET])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid or missing numeric value on CSV line {line_number}"
                ) from error
            if not all(row[name].strip() for name in CATEGORICAL_FEATURES):
                raise ValueError(f"Missing categorical value on CSV line {line_number}")
            features.append(feature)
            targets.append(target)

    if len(features) < 5:
        raise ValueError("CSV must contain at least 5 valid house records")
    return features, targets


def build_model() -> Pipeline:
    """Build a preprocessing and regression pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_evaluate(
    features: list[dict[str, Any]], targets: list[float]
) -> tuple[Pipeline, dict[str, float]]:
    """Fit on a training split and return held-out regression metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "house_prices.csv",
        help="CSV file containing the features and target",
    )
    args = parser.parse_args()

    features, targets = load_dataset(args.data)
    model, metrics = train_and_evaluate(features, targets)
    sample_home = {
        "area_sqft": 1_800,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 10,
        "distance_to_city_km": 12,
        "parking_spaces": 2,
        "location": "suburban",
        "condition": "good",
    }
    estimate = model.predict([sample_home])[0]

    print(f"Records: {len(features)}")
    print(f"Test MAE:  ${metrics['mae']:,.2f}")
    print(f"Test RMSE: ${metrics['rmse']:,.2f}")
    print(f"Test R^2:  {metrics['r2']:.3f}")
    print(f"Example estimated price: ${estimate:,.2f}")
    print("Note: metrics and estimate are based on synthetic data.")


if __name__ == "__main__":
    main()
