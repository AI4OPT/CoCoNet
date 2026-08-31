import numpy as np

from learnscore.calibrate import conformal_threshold


def test_order_statistic_matches_hand_computed_example():
    # n=9, alpha=0.1 -> k = ceil(10*0.9) = 9 -> the largest score
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert conformal_threshold(scores, alpha=0.1) == 9.0


def test_infinite_when_k_exceeds_n():
    scores = np.array([1.0, 2.0, 3.0])
    # n=3 -> k = ceil(4*0.99) = 4 > n
    assert conformal_threshold(scores, alpha=0.01) == float("inf")


def test_threshold_is_a_member_of_the_calibration_set():
    rng = np.random.default_rng(0)
    scores = rng.exponential(size=50)
    q = conformal_threshold(scores, alpha=0.1)
    assert q in scores
