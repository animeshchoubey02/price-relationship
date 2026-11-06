# price-ratio

Learning relative price relationships across retail products using product attributes, pairwise modeling, and graph-based reconciliation.

## Problem

Product-level pricing signals are uneven. New, low-volume, seasonal, or infrequently repriced products may have limited direct evidence for positioning. Related products can provide useful information: differences in brand, category, specifications, dimensions, weight, and product descriptions can help explain relative price differences.

This project models the log price ratio between product pairs rather than fitting an independent absolute-price prediction for every item. Pairwise estimates are then reconciled over a product graph to obtain one globally consistent relative-price structure.

## Data

The included example catalog is derived from the open product-catalog files supplied with this project. It contains product-level observations and observed selling prices. The accompanying category reference preserves the supplied category hierarchy. Marketplace URLs and source row identifiers are not carried into the normalized example catalog; product identifiers are replaced with neutral catalog keys.

The example is intentionally small (63 products), so it is suitable for inspecting the workflow and running the code, not for drawing broad conclusions about retail pricing. The raw input workbooks were Excel files despite their `.csv` filenames; the repository includes normalized CSV files for straightforward use.

## Method

1. Normalize product records and observed prices.
2. Construct product pairs and attribute-difference features.
3. Learn `log(price_i / price_j)` from observed prices.
4. Predict pairwise log ratios for the product graph.
5. Reconcile edge estimates with weighted least squares into latent log prices.
6. Relearn the attribute-to-ratio mapping from graph-implied relationships, then reconcile again.
7. Optionally incorporate trusted price-ratio anchors supplied by the user.

```text
product catalog
      │
      ▼
attribute-based product pairs ── observed log price ratios
      │
      ▼
pairwise relationship model
      │
      ▼
predicted graph edge ratios ◄── optional trusted anchors
      │
      ▼
graph reconciliation → corrected relationships
      │
      ▼
relearn pairwise model → reconcile again
      │
      ▼
globally consistent relative-price structure
```

## Run

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python -m price_ratio
pytest
```

To use another product catalog:

```bash
python -m price_ratio path/to/catalog.csv --max-pairs 20000
```

Required columns are `product_id`, `price`, and `category`. Optional columns used when present include `brand`, `product_name`, `description`, `specification`, `technical_details`, `shipping_weight_g`, and `product_dimensions`. Missing optional fields are treated as empty.

## Evaluation

The command reports held-out pairwise MAE and R² in log-ratio space, graph edge RMSE after reconciliation, and a descriptive centered log-price RMSE. Pairwise train/test splitting is used for this compact demonstration; pairs can share products across splits, so the held-out metrics should not be interpreted as a strict unseen-product generalization estimate. For a larger application, use product-disjoint or family-disjoint evaluation and temporal validation.

## Important design choices

- **Relative rather than absolute target:** log ratios encode directional and proportional price differences.
- **Attribute differences:** pair features represent similarities and differences between products, rather than arbitrary row identity.
- **Target-free feature construction:** categorical and text features are not ordered using observed prices.
- **Graph reconciliation:** local pairwise predictions are projected onto a globally realizable price vector.
- **Scale identification:** ratios identify prices only up to a common multiplicative constant; the solver fixes the mean log price to zero.
- **Anchors are explicit:** trusted relationships are optional inputs, not inferred or fabricated by the demo.

## Limitations

The example catalog is small and heterogeneous. Pair construction currently compares all available products (subject to a pair cap); a production-quality system should define candidate pairs using category-aware retrieval or domain-specific comparability rules. Text overlap is a lightweight baseline, not a semantic product-matching model. The graph objective assumes pairwise log ratios are additive and does not itself establish that the learned structure is commercially correct.

A globally consistent graph can still be globally wrong. Validation should include anchor sensitivity, family-level holdouts, edge coverage, cycle consistency, and review of the resulting relative-price structure.

## Public scope

This is a public implementation of a generalizable relative-pricing approach, built around the supplied open product-catalog observations. It is not a copy of a proprietary production system and contains no employer-specific data, schemas, business rules, infrastructure, or results.

## Extensions

- Category-aware candidate-pair generation and family-disjoint evaluation.
- Richer attribute parsing and unit normalization.
- Calibrated edge confidence and robust losses.
- Anchor sensitivity analysis and graph diagnostics.
- Iterative learn → reconcile → relearn workflow with explicit convergence criteria.
