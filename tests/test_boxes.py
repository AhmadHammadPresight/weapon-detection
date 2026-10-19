import pytest

from wd.boxes import area, clip, iou, parse_yolo_line, yolo_to_xyxy


def test_yolo_centre_and_size_become_pixel_corners():
    assert yolo_to_xyxy(0.5, 0.5, 0.2, 0.1, 1000, 800) == pytest.approx((400, 360, 600, 440))


def test_yolo_uses_width_for_x_and_height_for_y():
    assert yolo_to_xyxy(0.25, 0.75, 0.5, 0.5, 200, 100) == pytest.approx((0, 50, 100, 100))


def test_yolo_box_past_the_edge_is_clipped_to_the_image():
    assert yolo_to_xyxy(0.95, 0.05, 0.2, 0.2, 100, 100) == pytest.approx((85, 0, 100, 15))


def test_yolo_dataset_label_converts_to_the_known_pixel_box():
    # handgun_robbery_var10_img1.txt, class 2 (gun), in a 1024 x 1024 image
    _, cx, cy, w, h = parse_yolo_line("2 0.532973 0.364956 0.113959 0.068032")
    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, 1024, 1024)
    assert (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)) == (487.4, 338.9, 604.1, 408.5)


@pytest.mark.parametrize("values", [(-0.1, 0.5, 0.1, 0.1), (0.5, 1.2, 0.1, 0.1), (0.5, 0.5, 1.5, 0.1)])
def test_yolo_values_outside_0_1_are_rejected(values):
    with pytest.raises(ValueError):
        yolo_to_xyxy(*values, 100, 100)


def test_yolo_needs_a_real_image_size():
    with pytest.raises(ValueError):
        yolo_to_xyxy(0.5, 0.5, 0.1, 0.1, 0, 100)


def test_parse_yolo_line_needs_five_fields():
    assert parse_yolo_line("3 0.1 0.2 0.3 0.4") == (3, 0.1, 0.2, 0.3, 0.4)
    with pytest.raises(ValueError):
        parse_yolo_line("3 0.1 0.2 0.3")


def test_clip_and_area():
    assert clip((-5, -5, 50, 150), 40, 100) == (0, 0, 40, 100)
    assert area((0, 0, 10, 5)) == 50
    assert area((10, 10, 5, 5)) == 0


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0
