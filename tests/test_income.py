import numpy as np
import pytest

from fa import income as I, requirements as Rq

GIG = [30000, 32000, 18000]


def test_stats_and_isi_match_paper_worked_example():
    st = I.stability(GIG)
    assert st["cv"] == pytest.approx(0.28394541729, rel=1e-9)
    assert st["semi_dev_ratio"] == pytest.approx(0.25980762113, rel=1e-9)
    assert st["drought_frac"] == pytest.approx(1 / 3)
    assert st["isi"] == 45.1 and st["band"] == "volatile"


def test_shrinkage_fit():
    m = I.fit(GIG)
    assert m["shrink_weight"] == pytest.approx(4 / 7)
    assert m["phi"] == pytest.approx(0.3)
    assert m["sigma"] == pytest.approx(0.27981523723, rel=1e-9)
    m12 = I.fit([52000, 48000, 61000, 70000, 66000, 40000, 35000, 38000, 55000, 72000, 68000, 45000])
    assert m12["shrink_weight"] == pytest.approx(0.25)
    assert m12["sigma"] == pytest.approx(0.23889485673, rel=1e-9)
    assert m12["phi"] == pytest.approx(0.42581798383, rel=1e-9)


def test_simulation_level_correction():
    paths = I.simulate(I.fit(GIG))
    assert paths.shape == (I.N_PATHS, I.HORIZON)
    assert paths.mean() == pytest.approx(np.mean(GIG), rel=1e-9)


def test_ruin_monotone_and_safe_emi_is_boundary():
    paths = I.simulate(I.fit(GIG))
    rs = [I.ruin_probability(paths, 21000, e, 25000) for e in (0, 1000, 3000, 6000, 10000)]
    assert all(a <= b for a, b in zip(rs, rs[1:]))
    s = I.safe_emi(paths, 21000, 25000, 0.10)
    assert I.ruin_probability(paths, 21000, s, 25000) <= 0.10
    assert I.ruin_probability(paths, 21000, s * 1.05 + 1, 25000) > 0.10
    # paper worked example: about 3,200 at nominal alpha
    assert 2500 < s < 4000


def test_required_buffer_achieves_alpha():
    paths = I.simulate(I.fit(GIG))
    b = I.required_buffer(paths, 21000, 2000, 0.10)
    assert I.ruin_probability(paths, 21000, 2000, b + 1e-6) <= 0.10


def test_calibrated_alpha_table():
    assert I.calibrated_alpha(3) == 0.005
    assert I.calibrated_alpha(12) == 0.01
    assert I.calibrated_alpha(30) == 0.03


def test_emi_formula():
    assert I.emi_amount(300000, 12, 36) == pytest.approx(9964.2929438, rel=1e-9)
    assert I.emi_amount(1200, 0, 12) == 100
    assert I.emi_amount(0, 12, 12) == 0


def test_scale_invariance():
    a = I.analyse(GIG, 21000, 5000, 25000)
    b = I.analyse([x / 1000 for x in GIG], 21, 5, 25)
    assert a["ruin_probability"] == b["ruin_probability"]
    assert a["stability"]["isi"] == b["stability"]["isi"]


def test_checklist_unknowns_not_inferred():
    r = I.analyse(GIG, 21000, 9964.29, 25000)
    items = {i["key"]: i for i in Rq.checklist(GIG, 21000, 3000, 9964.29, 25000, r["stability"],
                                               r["ruin_probability"], r["safe_new_emi"], r["required_buffer"])}
    assert items["consistency"]["status"] == "unknown"
    assert items["credit_info"]["status"] == "unknown"
    assert items["repayment"]["value"] == "-12,964"
    assert items["cashflow"]["value"] == "2 of 3 months short"
    assert items["reserve"]["value"] == "1.19 months"
