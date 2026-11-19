from pathlib import Path
import pandas as pd

TRAIN_INPUT = Path("data/processed/train.csv")
TEST_INPUT = Path("data/processed/test.csv")

X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")

FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
TARGET = "Species"


def transform_split(input_path):
    data = pd.read_csv(input_path)

    missing = sorted(set(FEATURE_COLUMNS + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {input_path}: {missing}")

    # For this classic Iris dataset, we do not invent new mathematical
    # features. We explicitly select the four measured features and separate
    # the target. Scaling is kept inside the sklearn Pipeline to prevent
    # cross-validation and train/test leakage.
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET].copy()

    return X, y


def main():
    X_train, y_train = transform_split(TRAIN_INPUT)
    X_test, y_test = transform_split(TEST_INPUT)

    X_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_frame(name=TARGET).to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_frame(name=TARGET).to_csv(Y_TEST_PATH, index=False)

    print("Feature selection and target separation completed.")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Saved split files under: {X_TRAIN_PATH.parent}")


if __name__ == "__main__":
    main()
