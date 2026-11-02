from app.grounding import ungrounded


def test_grounding_flags_invented_numbers_and_typoed_dates_but_not_real_ones():
    trace = [{"tool": "compare_periods", "result": {"mean_a": 1063.19, "mean_b": 1309.4, "pct_change": 23.2, "first_day": "2023-11-01"}}]
    q = "alert window 2023-11-01 to 2023-11-02 peak z-score 4.53"
    ok = "Spend rose from $1,063 to $1,309 (+23.2%) starting 2023-11-01 (z 4.53)."
    assert ungrounded(ok, trace, q) == []
    bad = ungrounded("It averaged $1,343/day and ended 2103-11-02.", trace, q)
    assert "1,343" in bad and "2103-11-02" in bad


def test_grounding_sees_through_unicode_hyphens_and_nbsp():
    trace = [{"tool": "x", "result": {"d": "2023-11-01"}}]
    assert ungrounded("began 2023‑11‑01", trace, "") == []          # U+2011 hyphens, grounded
    assert ungrounded("began 2103‑11‑02", trace, "") == ["2103-11-02"]  # typo caught even with U+2011
