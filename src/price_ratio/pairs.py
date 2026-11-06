"""Construct pairwise feature vectors and observed log-price-ratio targets."""

import re
from itertools import combinations

import numpy as np

pd = None


def _tokens(row, columns):
    text = " ".join(str(row.get(c, "")) for c in columns)
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _numeric(text):
    vals = re.findall(r"[0-9]+(?:\.[0-9]+)?", str(text))
    return np.asarray([float(x) for x in vals[:8]], dtype=float)


def pair_features(a, b):
    f = {
        "same_brand": float(a.brand.lower() == b.brand.lower()),
        "same_category": float(a.category == b.category),
        "same_category_leaf": float(
            a.category.split("|")[-1].strip() == b.category.split("|")[-1].strip()
        ),
        "category_depth_gap": abs(len(a.category.split("|")) - len(b.category.split("|"))),
        "name_token_jaccard": _jaccard(
            _tokens(a, ("product_name",)), _tokens(b, ("product_name",))
        ),
        "description_token_jaccard": _jaccard(
            _tokens(a, ("description", "specification", "technical_details")),
            _tokens(b, ("description", "specification", "technical_details")),
        ),
    }
    wa, wb = a.get("shipping_weight_g", np.nan), b.get("shipping_weight_g", np.nan)
    f["weight_abs_diff_g"] = abs(float(wa) - float(wb)) if _valid(wa) and _valid(wb) else 0.0
    f["weight_log_ratio"] = (
        float(np.log(float(wa) / float(wb)))
        if _valid(wa) and _valid(wb) and float(wa) > 0 and float(wb) > 0
        else 0.0
    )
    da, db = _numeric(a.get("product_dimensions", "")), _numeric(b.get("product_dimensions", ""))
    for i in range(3):
        x = da[i] if len(da) > i else np.nan
        y = db[i] if len(db) > i else np.nan
        f[f"dimension_{i + 1}_abs_diff"] = abs(x - y) if np.isfinite(x) and np.isfinite(y) else 0.0
        f[f"dimension_{i + 1}_log_ratio"] = (
            np.log(x / y) if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0 else 0.0
        )
    return f


def _valid(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def build_pairs(catalog, max_pairs=20000, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(catalog)
    all_pairs = list(combinations(range(n), 2))
    if len(all_pairs) > max_pairs:
        chosen = np.sort(rng.choice(len(all_pairs), max_pairs, replace=False))
        all_pairs = [all_pairs[i] for i in chosen]
    rows = []
    targets = []
    for i, j in all_pairs:
        a, b = catalog.iloc[i], catalog.iloc[j]
        rows.append(pair_features(a, b))
        targets.append(np.log(float(a.price) / float(b.price)))
    import pandas as pd

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], 0).fillna(0)
    return X, np.asarray(targets), np.asarray(all_pairs, dtype=int)
