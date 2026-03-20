from kswing_sentinel.cost_model import SessionCostModel


def test_cost_model_nonlinear_participation_and_liquidity_bucket():
    cm = SessionCostModel()
    low_part = cm.estimate("KRX", "CORE_DAY", participation=0.02, liquidity_bucket="high").total_bps
    high_part = cm.estimate("KRX", "CORE_DAY", participation=0.2, liquidity_bucket="low").total_bps
    assert high_part > low_part
