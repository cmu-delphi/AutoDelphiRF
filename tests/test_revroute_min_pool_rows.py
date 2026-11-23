"""RevRoute's minimum pool size merges undersized pools into a neighbour."""
import numpy as np

from autodelphirf.revroute.pooling.clustering import (SparseTaskGeometry,
                                                      enforce_minimum_pool_rows)


def geometry(tasks, profiles):
    n = len(tasks)
    return SparseTaskGeometry(tasks, np.asarray(profiles, float).reshape(n, -1),
                              [np.zeros(3)] * n, np.zeros(n))


def test_zero_minimum_keeps_the_similarity_pools():
    g = geometry([("us", 0), ("us", 7)], [0., 5.])
    assert enforce_minimum_pool_rows(g, [0, 1], [27, 27], 0).tolist() == [0, 1]


def test_single_location_pools_grow_along_adjacent_lags():
    tasks = [("us", 0), ("us", 7), ("us", 14), ("us", 21)]
    g = geometry(tasks, [0., 1., 2., 3.])
    labels = [0, 1, 2, 2]
    # Lag 0 (27 rows) can only reach lag 7, and 54 rows still fall short of 100.
    assert enforce_minimum_pool_rows(g, labels, [27] * 4, 100).tolist() == [0, 0, 0, 0]
    # At 50, lag 0 joins lag 7 and the 54-row pool of lags 14 and 21 is kept.
    assert enforce_minimum_pool_rows(g, labels, [27] * 4, 50).tolist() == [0, 0, 1, 1]


def test_an_undersized_pool_joins_its_nearest_neighbouring_pool():
    tasks = [("a", 0), ("a", 7), ("b", 0)]
    # (a, 0) is far closer to (b, 0) than to (a, 7).
    g = geometry(tasks, [0., 10., .1])
    assert enforce_minimum_pool_rows(g, [0, 1, 2], [20, 200, 200], 100).tolist() == [0, 1, 0]


def test_a_pool_without_structural_neighbours_is_left_alone():
    g = geometry([("a", 0), ("b", 7)], [0., 1.])
    assert enforce_minimum_pool_rows(g, [0, 1], [5, 5], 100).tolist() == [0, 1]


def test_learned_pools_apply_the_minimum(tmp_path, monkeypatch):
    import pandas as pd
    from types import SimpleNamespace
    import autodelphirf.revroute.pooling.plan as pooling_plan

    tasks = [("us", 0), ("us", 7), ("us", 14)]
    g = geometry(tasks, [0., 1., 2.])
    prototypes = SimpleNamespace(rows=[np.arange(27), np.arange(27), np.arange(60)])
    monkeypatch.setattr(pooling_plan, "rolling_completed_training",
                        lambda prepared, cutoff, days: prepared)
    monkeypatch.setattr(pooling_plan, "sparse_task_pooling",
                        lambda *args, **kwargs: (g, np.array([0, 1, 2]), .1, None, prototypes))
    schedule = pd.DataFrame({"test_date": pd.to_datetime(["2022-03-01"]),
                             "training_days": [189], "target_lag": [63],
                             "target_column": ["log_value_target_7dav"]})
    output = tmp_path / "pools.csv"
    pooling_plan.learn_task_pools(pd.DataFrame(), schedule, output, min_pool_rows=50)
    assert pd.read_csv(output).pool.tolist() == [0, 0, 1]
    summary = pd.read_csv(tmp_path / "revroute_pool_summary.csv")
    assert summary.min_pool_rows.tolist() == [50]
