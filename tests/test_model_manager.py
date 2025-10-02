from model.model_manager import train_technical_predictor, aggregate_decision


def test_train_predictor_smoke():
    predictor = train_technical_predictor("AAPL")
    assert predictor.model is not None


def test_aggregate_decision_outputs():
    decision, sent_score, tech_proba = aggregate_decision("AAPL", "Apple beats earnings estimates and raises guidance")
    assert decision in ("BUY", "SELL", "HOLD")
    assert 0.0 <= sent_score <= 1.0
    assert 0.0 <= tech_proba <= 1.0


