import dataclasses

import pytest

from wd.dataset import CLASSES, load_manifest, check_manifest, parse_name


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_committed_split_keeps_every_rule(manifest):
    assert check_manifest(manifest) == []


def test_split_sizes_and_balance(manifest):
    for split in ("train", "test"):
        samples = manifest.split(split)
        assert len(samples) == 20
        assert sorted(s.label for s in samples) == sorted([*CLASSES] * 10)


def test_no_scene_variant_is_in_both_splits(manifest):
    train = {s.variant for s in manifest.split("train")}
    test = {s.variant for s in manifest.split("test")}
    assert not train & test
    # one image per variant, so 40 different scenes
    assert len(train) == 20 and len(test) == 20


def test_each_image_has_one_weapon_box_of_its_scene_class(manifest):
    for s in manifest.samples:
        assert [t.label for t in s.boxes] == [s.label]


def test_images_sit_in_their_split_folder(manifest):
    for s in manifest.samples:
        assert s.file == f"images/{s.split}/{s.name}"


def test_check_catches_a_leaked_variant(manifest):
    train = manifest.split("train")[0]
    test = manifest.split("test")
    leaked = dataclasses.replace(test[0], variant=train.variant)
    broken = dataclasses.replace(manifest, samples=(*manifest.split("train"), leaked, *test[1:]))
    assert any("both train and test" in p for p in check_manifest(broken, check_files=False))


def test_check_catches_unbalanced_and_bad_boxes(manifest):
    s = manifest.split("test")[0]
    bad = dataclasses.replace(s, boxes=(dataclasses.replace(s.boxes[0], box=(10, 10, 5, 5)),))
    broken = dataclasses.replace(manifest, samples=(*manifest.split("train"), bad, *manifest.split("test")[2:]))
    problems = check_manifest(broken, check_files=False)
    assert any("test has 19 images" in p for p in problems)
    assert any("outside" in p for p in problems)


def test_check_catches_a_changed_image_file(manifest, tmp_path, monkeypatch):
    s = manifest.split("train")[0]
    changed = dataclasses.replace(s, sha256="0" * 64)
    broken = dataclasses.replace(manifest, samples=(changed, *manifest.samples[1:]))
    assert any("SHA-256" in p for p in check_manifest(broken))


def test_parse_name():
    assert parse_name("handgun_robbery_var10_img4.png") == ("handgun_robbery_var10", "gun")
    assert parse_name("knife_robbery_var3_img1.png") == ("knife_robbery_var3", "knife")
    with pytest.raises(ValueError):
        parse_name("knifes_Knife_scenario_25.png")
