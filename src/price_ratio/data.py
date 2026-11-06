"""Load and normalize a product-level retail catalog."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"product_id", "price", "category"}


def load_catalog(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["product_id"] = df.product_id.astype(str)
    df["price"] = pd.to_numeric(df.price, errors="coerce")
    df = df.dropna(subset=["product_id", "price", "category"])
    df = df[df.price > 0].drop_duplicates("product_id").reset_index(drop=True)
    for col in (
        "brand",
        "product_name",
        "description",
        "specification",
        "technical_details",
        "product_dimensions",
        "variants",
    ):
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if len(df) < 3:
        raise ValueError("At least three products with positive prices are required.")
    return df


def numeric_measure(text: str, pattern: str, factor: float = 1.0) -> float:
    m = re.search(pattern, str(text), flags=re.I)
    return float(m.group(1)) * factor if m else np.nan
