# Design decisions

## Learn ratios, not independent prices

The target is `log(price_i / price_j)`. This focuses the model on relative positioning and makes reciprocal relationships antisymmetric when pair order is reversed.

## Keep feature construction target-free

Categorical representation is based on product metadata, not category-level price statistics. Text overlap and attribute differences are computed from the product records.

## Reconcile local estimates globally

Independent edge predictions may violate cycle consistency. The graph stage finds the closest price vector under a weighted least-squares objective. This is a projection onto the space of realizable pairwise log ratios. A second model is then trained on graph-implied edge ratios, so the structural correction influences the learned attribute relationship rather than remaining only an inference-time adjustment.

## Do not invent trusted anchors

Anchors are an optional explicit input to `reconcile`; the example does not silently designate observed pairs as trusted relationships. Anchor selection requires domain justification and sensitivity analysis.

## Evaluation caveat

The compact demonstration uses pair-level random splitting. Since products can occur in both train and test pairs, this is not an unseen-product test. A serious evaluation should split by product/family or time and should assess structural quality in addition to predictive error.

## Deliberate omissions

The public implementation omits proprietary candidate-generation rules, business-defined price structures, internal feature schemas, and production infrastructure. It keeps the generalizable learning and graph formulation.
