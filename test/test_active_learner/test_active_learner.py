from spark_matcher.activelearner.active_learner import ScoringLearner
from spark_matcher.scorer.scorer import Scorer


def test__get_uncertainty_improvement(spark_session):
    scorer = Scorer(spark_session)
    myScoringLearner = ScoringLearner(col_names=[''], scorer=scorer, n_uncertainty_improvement=5)
    myScoringLearner.uncertainties = [0.4, 0.2, 0.1, 0.08, 0.05, 0.03]
    assert myScoringLearner._get_uncertainty_improvement() == 0.2


def test__is_converged(spark_session):
    scorer = Scorer(spark_session)
    myScoringLearner = ScoringLearner(col_names=[''], scorer=scorer, min_nr_samples=5, uncertainty_threshold=0.1,
                                      uncertainty_improvement_threshold=0.01, n_uncertainty_improvement=5)

    # insufficient labelled samples
    myScoringLearner.uncertainties = [0.4, 0.39, 0.395]
    myScoringLearner.counter_total = len(myScoringLearner.uncertainties)
    assert not myScoringLearner._is_converged()

    # too large improvement in last 5 iterations
    myScoringLearner.uncertainties = [0.4, 0.2, 0.19, 0.18, 0.17, 0.16]
    myScoringLearner.counter_total = len(myScoringLearner.uncertainties)
    assert not myScoringLearner._is_converged()

    # improvement in last 5 iterations below threshold and sufficient labelled cases
    myScoringLearner.uncertainties = [0.19, 0.1, 0.08, 0.05, 0.03, 0.02]
    myScoringLearner.counter_total = len(myScoringLearner.uncertainties)
    assert myScoringLearner._is_converged()
