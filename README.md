# House Price Prediction

A small, reproducible Python project that trains a house-price regression model,
reports held-out evaluation metrics, and estimates a price for a sample home.
The included dataset is synthetic and is intended for learning and demonstration;
it does not represent actual listings or guarantee accuracy in a real market.

## Project files

- `data/house_prices.csv` — 1,000 generated example homes.
- `generate_data.py` — reproducibly regenerates the synthetic dataset.
- `house_price_prediction.py` — model training, evaluation, and prediction.
- `requirements.txt` — Python package dependencies.

## Requirements

- Python 3.9 or later
- pip

## Setup

Run these commands from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate

```powershell
python house_price_prediction.py
```

The script uses a fixed random seed and a held-out test split. It prints mean
absolute error (MAE), root mean squared error (RMSE), and R-squared (R²), followed
by an example prediction. MAE and RMSE are in the same currency units as the
dataset (illustrative dollars). R² is a score, not a percentage accuracy.

You can provide a different CSV with the same columns:

```powershell
python house_price_prediction.py --data path\to\your_houses.csv
```

The CSV must include the feature columns listed below and a numeric
`price` target column. Keep all records from the same home in only one split if
you adapt this example to repeated sales or time-based data. For real-world use,
evaluate on representative, recent transactions and account for location,
currency, data quality, and market changes.

## Regenerate the example dataset

```powershell
python generate_data.py
```

The generator uses a fixed seed, so it produces the same data each time.
To create a larger dataset, pass a row count:

```powershell
python generate_data.py --rows 2000
```

## Features

| Column | Description |
| --- | --- |
| `area_sqft` | Interior floor area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `age_years` | Approximate age of the home |
| `location` | Example location category |
| `condition` | `fair`, `good`, or `excellent` |
| `distance_to_city_km` | Distance to the city center in kilometers |
| `parking_spaces` | Number of parking spaces |
| `price` | Synthetic target price |

This project is a teaching baseline, not a valuation tool. A model trained on
synthetic data learns the generator's assumptions rather than actual housing
market behavior.
