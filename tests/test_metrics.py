import pytest

from wd.metrics import GroundTruth, Pred, average_precision, choose_class_thresholds, choose_threshold, counts_at, match, score

TRUTH = [GroundTruth("a", "gun", (0, 0, 10, 10)), GroundTruth("b", "gun", (0, 0, 10, 10))]


def test_match_needs_the_right_image_class_and_overlap():
    preds = [
        Pred("a", "gun", 0.9, (0, 0, 10, 10)),  # hit
        Pred("a", "gun", 0.8, (0, 0, 10, 10)),  # duplicate: the box is already claimed
        Pred("b", "knife", 0.7, (0, 0, 10, 10)),  # wrong class
        Pred("b", "gun", 0.6, (8, 8, 18, 18)),  # too little overlap
    ]
    assert match(preds, TRUTH) == [True, False, False, False]


def test_counts_and_ratios_at_a_threshold():
    preds = [Pred("a", "gun", 0.9, (0, 0, 10, 10)), Pred("b", "gun", 0.4, (0, 0, 10, 10)), Pred("b", "gun", 0.95, (50, 50, 60, 60))]
    c = counts_at(preds, TRUTH, 0.5)
    assert (c.tp, c.fp, c.fn) == (1, 1, 1)
    assert (c.precision, c.recall) == (0.5, 0.5)
    c = counts_at(preds, TRUTH, 0.3)
    assert (c.tp, c.fp, c.fn) == (2, 1, 0)


def test_average_precision():
    perfect = [Pred("a", "gun", 0.9, (0, 0, 10, 10)), Pred("b", "gun", 0.8, (0, 0, 10, 10))]
    assert average_precision(perfect, TRUTH) == 1
    # a false alarm ranked first: precision 1/2 at recall 1/2, then 2/3 at recall 1
    ranked = [Pred("b", "gun", 0.99, (50, 50, 60, 60)), *perfect]
    assert average_precision(ranked, TRUTH) == pytest.approx(0.5 * 2 / 3 + 0.5 * 2 / 3)
    assert average_precision([], TRUTH) == 0
    assert average_precision(perfect, []) == 0


def test_choose_threshold_takes_the_middle_of_the_best_range():
    preds = [Pred("a", "gun", 0.62, (0, 0, 10, 10)), Pred("b", "gun", 0.71, (0, 0, 10, 10)), Pred("b", "gun", 0.22, (50, 50, 60, 60))]
    # F1 is best (1.0) for thresholds 0.25 .. 0.60
    assert choose_threshold(preds, TRUTH) == 0.40


def test_score_reports_per_class_and_overall():
    truths = [*TRUTH, GroundTruth("c", "knife", (0, 0, 10, 10))]
    preds = [Pred("a", "gun", 0.9, (0, 0, 10, 10)), Pred("c", "knife", 0.9, (0, 0, 10, 10))]
    s = score(preds, truths, ("gun", "knife"), 0.5)
    assert s["per_class"]["gun"]["recall"] == 0.5
    assert s["per_class"]["knife"]["ap50"] == 1
    assert s["overall"]["tp"] == 2 and s["overall"]["fn"] == 1
    assert s["overall"]["map50"] == 0.75


def test_class_thresholds_are_chosen_per_class():
    truths = [GroundTruth("a", "gun", (0, 0, 10, 10)), GroundTruth("c", "knife", (0, 0, 10, 10))]
    preds = [
        Pred("a", "gun", 0.3, (0, 0, 10, 10)),  # a weak but right gun
        Pred("c", "knife", 0.9, (0, 0, 10, 10)),
        Pred("neg", "knife", 0.6, (0, 0, 10, 10)),  # a knife on a weapon-free image
    ]
    t = choose_class_thresholds(preds, truths, ("gun", "knife"))
    assert t["gun"] <= 0.3 < 0.6 < t["knife"] <= 0.9
    s = score(preds, truths, ("gun", "knife"), t)
    assert s["thresholds"] == t
    assert (s["overall"]["tp"], s["overall"]["fp"]) == (2, 0)
    assert counts_at(preds, truths, 0.5).fp == 1  # one shared threshold lets the false knife through
