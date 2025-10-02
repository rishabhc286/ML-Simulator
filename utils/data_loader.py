import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_sample_regression():
    """
    Load the Sample Regression dataset (data/sample_regression.csv).
    Returns a DataFrame with columns: Feature, Target
    """
    path = DATA_DIR / "sample_regression.csv"
    return pd.read_csv(path)