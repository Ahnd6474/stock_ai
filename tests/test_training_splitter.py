from datetime import date, timedelta

from kswing_sentinel.training import WalkForwardSplitter


def test_walk_forward_builds_folds():
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(200)]
    folds = WalkForwardSplitter().build(days, train_size=100, valid_size=20, step=20)
    assert len(folds) >= 3
    assert folds[0].train_start < folds[0].valid_start
