from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/Iris.csv")
OUTPUT_PATH = Path("data/ingested/iris.csv")

REQUIRED_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
    "Species",
]


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}. Put the Iris CSV at this location."
        )

    data = pd.read_csv(RAW_PATH)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep only model-relevant columns. The uploaded CSV has an identifier
    # column named 'Id'; it is not a predictive feature.
    data = data[REQUIRED_COLUMNS].copy()

    if data.empty:
        raise ValueError("The input dataset is empty.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Data ingestion completed: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
