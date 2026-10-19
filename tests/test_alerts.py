from datetime import datetime

from wd.alerts import AlertGrouper, alert_time, alerting, near
from wd.detectors import Detection, suppress_overlaps

GUN = Detection("gun", 0.8, (10, 10, 50, 40))
WEAK_GUN = Detection("gun", 0.3, (12, 12, 50, 40))
KNIFE = Detection("knife", 0.6, (100, 100, 140, 120))


def test_only_detections_at_or_over_the_threshold_alert():
    assert alerting([WEAK_GUN, GUN, KNIFE], 0.6) == [GUN, KNIFE]
    assert alerting([WEAK_GUN], 0.5) == []


def test_a_still_image_raises_one_alert_per_class():
    grouper = AlertGrouper(0.5, cooldown_seconds=10)
    changes = grouper.add(0.0, [GUN, KNIFE, WEAK_GUN])
    assert [(c.kind, c.alert.label) for c in changes] == [("new", "gun"), ("new", "knife")]
    assert grouper.alerts[0].detections == [GUN, KNIFE]  # the boxed frame shows every alerting weapon


def test_a_frame_below_the_threshold_raises_nothing():
    grouper = AlertGrouper(0.5, cooldown_seconds=10)
    assert grouper.add(0.0, [WEAK_GUN]) == []
    assert grouper.alerts == []


def test_repeated_frames_within_the_cooldown_join_one_alert():
    grouper = AlertGrouper(0.5, cooldown_seconds=10)
    grouper.add(0.0, [GUN])
    grouper.add(0.5, [GUN])
    grouper.add(9.0, [GUN])
    assert len(grouper.alerts) == 1
    assert grouper.alerts[0].frames == 3
    assert (grouper.alerts[0].first_seen, grouper.alerts[0].last_seen) == (0.0, 9.0)


def test_the_cooldown_counts_from_the_last_sighting():
    grouper = AlertGrouper(0.5, cooldown_seconds=10)
    for t in (0, 8, 16, 24):
        grouper.add(float(t), [GUN])
    assert len(grouper.alerts) == 1


def test_a_gap_longer_than_the_cooldown_opens_a_new_alert():
    grouper = AlertGrouper(0.5, cooldown_seconds=10)
    grouper.add(0.0, [GUN])
    changes = grouper.add(10.5, [GUN])
    assert [c.kind for c in changes] == ["new"]
    assert [a.key for a in grouper.alerts] == [0, 1]


def test_a_stronger_frame_replaces_the_alert_frame():
    grouper = AlertGrouper(0.5, cooldown_seconds=10)
    first = Detection("gun", 0.6, (0, 0, 10, 10))
    stronger = Detection("gun", 0.9, (1, 1, 11, 11))
    grouper.add(1.0, [first])
    changes = grouper.add(2.0, [stronger])
    assert [c.kind for c in changes] == ["update"]
    alert = grouper.alerts[0]
    assert (alert.confidence, alert.best_at, alert.detections) == (0.9, 2.0, [stronger])
    assert grouper.add(3.0, [first]) == []  # a weaker frame keeps the stronger one


def test_classes_are_grouped_separately():
    grouper = AlertGrouper(0.5, cooldown_seconds=10)
    grouper.add(0.0, [GUN])
    changes = grouper.add(1.0, [KNIFE])
    assert [(c.kind, c.alert.label) for c in changes] == [("new", "knife")]


def test_alert_time_adds_the_offset_to_the_camera_start():
    assert alert_time(datetime(2026, 10, 19, 23, 59, 50), 12.5) == datetime(2026, 10, 20, 0, 0, 2, 500000)


def test_overlapping_boxes_keep_the_strongest_whatever_the_class():
    gun = Detection("gun", 0.7, (0, 0, 10, 10))
    knife_same_place = Detection("knife", 0.4, (1, 1, 10, 10))
    elsewhere = Detection("knife", 0.3, (50, 50, 60, 60))
    assert suppress_overlaps([knife_same_place, gun, elsewhere]) == [gun, elsewhere]


def test_per_class_thresholds():
    assert alerting([GUN, KNIFE, WEAK_GUN], {"gun": 0.25, "knife": 0.7}) == [GUN, WEAK_GUN]


# Persistence: in a video a weapon must be seen in min_frames consecutive frames, near where it was before.

def at(box_shift: float, confidence: float = 0.8, label: str = "knife") -> Detection:
    return Detection(label, confidence, (100 + box_shift, 100, 140 + box_shift, 120))


def test_no_alert_until_seen_in_enough_consecutive_frames():
    grouper = AlertGrouper(0.5, cooldown_seconds=10, min_frames=3)
    assert grouper.add(0.0, [at(0)]) == []
    assert grouper.add(0.5, [at(5)]) == []
    changes = grouper.add(1.0, [at(10)])
    assert [c.kind for c in changes] == ["new"]
    alert = grouper.alerts[0]
    assert (alert.first_seen, alert.best_at) == (0.0, 1.0)  # starts with the streak, shows the confirming frame


def test_a_missed_frame_breaks_the_streak():
    grouper = AlertGrouper(0.5, cooldown_seconds=10, min_frames=3)
    for t, dets in [(0.0, [at(0)]), (0.5, [at(0)]), (1.0, []), (1.5, [at(0)]), (2.0, [at(0)])]:
        assert grouper.add(t, dets) == []
    assert grouper.alerts == []


def test_a_frame_below_the_threshold_also_breaks_the_streak():
    grouper = AlertGrouper(0.5, cooldown_seconds=10, min_frames=3)
    for t, conf in [(0.0, 0.8), (0.5, 0.3), (1.0, 0.8), (1.5, 0.8)]:
        grouper.add(t, [at(0, conf)])
    assert grouper.alerts == []


def test_detections_jumping_around_the_frame_are_not_one_weapon():
    grouper = AlertGrouper(0.5, cooldown_seconds=10, min_frames=3)
    for t, shift in [(0.0, 0), (0.5, 400), (1.0, 0), (1.5, 400)]:
        grouper.add(t, [at(shift)])
    assert grouper.alerts == []


def test_a_different_class_in_the_same_place_does_not_continue_the_streak():
    grouper = AlertGrouper(0.5, cooldown_seconds=10, min_frames=2)
    grouper.add(0.0, [at(0, label="gun")])
    assert grouper.add(0.5, [at(0, label="knife")]) == []


def test_nearby_boxes_in_one_frame_are_one_weapon_and_one_alert():
    grouper = AlertGrouper(0.5, cooldown_seconds=10, min_frames=2)
    grouper.add(0.0, [at(0, 0.9), at(8, 0.6)])
    changes = grouper.add(0.5, [at(2, 0.9), at(10, 0.6)])
    assert len(changes) == 1 and len(grouper.alerts) == 1
    assert grouper.alerts[0].detections == [at(2, 0.9)]


def test_two_weapons_far_apart_are_tracked_separately_but_alert_once_per_class():
    grouper = AlertGrouper(0.5, cooldown_seconds=10, min_frames=2)
    grouper.add(0.0, [at(0), at(500)])
    changes = grouper.add(0.5, [at(0), at(500)])
    assert len(changes) == 1
    assert len(grouper.alerts[0].detections) == 2  # both boxes on the alert frame


def test_unconfirmed_sightings_keep_an_open_alert_going():
    grouper = AlertGrouper(0.5, cooldown_seconds=2, min_frames=2)
    grouper.add(0.0, [at(0)])
    grouper.add(0.5, [at(0)])  # confirmed: alert opens
    # flickers every other frame for a while: never confirmed again, but the same event
    for i in range(1, 12):
        grouper.add(0.5 + i, [at(0)] if i % 2 else [])
    grouper.add(12.0, [at(0)])
    grouper.add(12.5, [at(0)])  # confirmed again, still within 2 s of the last sighting
    assert len(grouper.alerts) == 1


def test_min_frames_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        AlertGrouper(0.5, 10, min_frames=0)


def test_near():
    box = (0, 0, 40, 20)
    assert near(box, (30, 0, 70, 20))  # moved by less than its length
    assert not near(box, (100, 0, 140, 20))
