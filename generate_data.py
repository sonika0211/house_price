"""Generate a reproducible synthetic house-price dataset."""

import argparse
import csv
import random
from pathlib import Path


FIELDS = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "location",
    "condition",
    "distance_to_city_km",
    "parking_spaces",
    "price",
]

LOCATION_PREMIUM = {
    "suburban": 25_000,
    "urban": 75_000,
    "rural": -20_000,
}
CONDITION_PREMIUM = {"fair": -15_000, "good": 0, "excellent": 30_000}


def generate_rows(row_count: int, seed: int = 42) -> list[dict[str, object]]:
    """Create plausible example records with noise and a known price formula."""
    if row_count < 1:
        raise ValueError("row_count must be at least 1")

    rng = random.Random(seed)
    rows = []
    for _ in range(row_count):
        area = rng.randint(650, 3_500)
        bedrooms = max(1, min(6, round(area / 550 + rng.uniform(-0.8, 1.0))))
        bathrooms = max(1, min(5, round(bedrooms * 0.7 + rng.uniform(-0.5, 1.0))))
        age = rng.randint(0, 80)
        location = rng.choices(
            ["suburban", "urban", "rural"], weights=[0.5, 0.3, 0.2]
        )[0]
        condition = rng.choices(
            ["fair", "good", "excellent"], weights=[0.2, 0.55, 0.25]
        )[0]
        distance = round(rng.uniform(1, 45), 1)
        parking = rng.randint(0, 3)

        price = (
            55_000
            + area * 185
            + bedrooms * 9_000
            + bathrooms * 13_000
            + LOCATION_PREMIUM[location]
            + CONDITION_PREMIUM[condition]
            - age * 850
            - distance * 1_100
            + parking * 6_000
            + rng.gauss(0, 18_000)
        )
        rows.append(
            {
                "area_sqft": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "age_years": age,
                "location": location,
                "condition": condition,
                "distance_to_city_km": distance,
                "parking_spaces": parking,
                "price": round(max(40_000, price), 2),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000, help="number of records")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "house_prices.csv",
        help="destination CSV file",
    )
    args = parser.parse_args()

    rows = generate_rows(args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic records to {args.output}")


if __name__ == "__main__":
    main()
