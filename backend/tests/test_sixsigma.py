import numpy as np
from app.sixsigma import capability, imr, nelson


def test_imr_known_values():
    s = imr([10, 12, 11, 13, 14])  # MR = 2,1,2,1 -> MRbar 1.5
    assert abs(s["sigma"] - 1.5 / 1.128) < 1e-9
    assert abs(s["center"] - 12) < 1e-9
    assert abs(s["ucl"] - (12 + 3 * 1.5 / 1.128)) < 1e-9


def test_rule1_spike():
    x = np.r_[np.random.default_rng(0).normal(100, 1, 40), 130]
    s = imr(x[:40])
    assert 40 in nelson(x, s["center"], s["sigma"])[1]


def test_slow_leak_trips_2_and_3_not_1():
    base = np.random.default_rng(1).normal(0, 1, 40)
    x = np.r_[base, np.linspace(0.3, 2.4, 8)]  # drift well inside 3σ
    s = imr(base)
    h = nelson(x, s["center"], s["sigma"])
    assert h[3] and h[2] and not [i for i in h[1] if i >= 40]


def test_rule2_nine_same_side():
    x = [-1, 1] * 5 + [1] * 9
    h = nelson(x, 0, 1)
    assert h[2] and h[2][0] == 17  # 9th point of the run of 1s starting at idx 9


def test_rule3_six_trending():
    h = nelson([5, 1, 2, 3, 4, 5, 6, 0], 3, 10)
    assert h[3] == [6]  # idx 1..6 = 6 rising points


def test_capability_over_budget():
    assert capability([10, 12, 11, 13, 14], usl=11)["cpk"] < 1
