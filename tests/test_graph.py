import numpy as np

from price_ratio.graph import reconcile


def test_reconcile_recovers_consistent_relative_prices():
    truth = np.log([10.0, 20.0, 40.0])
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    ratios = truth[edges[:, 0]] - truth[edges[:, 1]]
    got = reconcile(3, edges, ratios)
    assert np.allclose(got - got.mean(), truth - truth.mean(), atol=1e-5)
