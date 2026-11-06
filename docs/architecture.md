# Architecture

## Components

- `data.py`: validates the product-level catalog and normalizes required fields.
- `pairs.py`: creates pairwise attribute-difference features and observed log-price-ratio targets.
- `pipeline.py`: trains the pairwise estimator, evaluates held-out pairs, predicts graph edges, and invokes reconciliation.
- `graph.py`: solves a weighted least-squares system for one latent log price per product.
- `__main__.py`: command-line entry point.

## Data flow

A row represents one product. The pair builder creates an edge `(i, j)` and target `log(price_i / price_j)`. Features include brand/category agreement, category-depth difference, token overlap, shipping-weight differences/ratios, and parsed dimension differences/ratios.

The first model predicts one log ratio per edge. The graph solver finds product log prices `x` minimizing the weighted residual between each predicted edge and `x_i - x_j`. Those reconciled edge ratios become the targets for a second model fit, after which a second graph reconciliation produces the final relative-price structure. An optional anchor adds a weighted constraint on an individual product's log price.

## Interfaces and assumptions

Input is a CSV with `product_id`, `price`, and `category`; additional product attributes are optional. Prices must be positive. Pair orientation is stable: the target is always `log(price_i / price_j)`. The graph is identifiable only up to a common additive shift in log space, so a mean-zero constraint fixes scale.

The current example uses equal edge weights. The API accepts weights for future confidence-aware reconciliation.
