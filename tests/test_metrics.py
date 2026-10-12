import pandas as pd, pytest
from src.metrics import weighted_kappa, bootstrap_mean_ci
from src.pipeline import validate_ratings

def test_perfect_kappa():
    assert weighted_kappa([1,2,3],[1,2,3]) == pytest.approx(1.0)

def test_bootstrap_interval_contains_mean():
    m,lo,hi=bootstrap_mean_ci([2,3,4,3],iterations=300,seed=1)
    assert lo<=m<=hi

def test_invalid_rating_rejected():
    with pytest.raises(ValueError):
        validate_ratings(pd.DataFrame({"prompt_id":["p"],"rater_id":["a"],"rating":[6]}))
