import numpy as np
import pytest

from warehouse_vision_safety.safety_logic import (
    DepthValidationError,
    classify_distance,
    median_depth_in_bbox
)

@pytest.mark.parametrize(
    ("distance", "expected_state"),
    [
        (0.8, "STOP"),
        (1.0, "STOP"),
        (2.5, "SLOW"),
        (3.0, "SLOW"),
        (4.0, "CLEAR")
    ]
)
def test_classify_distance(distance, expected_state):
    assert classify_distance(distance, 1.0, 3.0) == expected_state

@pytest.mark.parametrize(
    "invalid_distance",
    [0.0, -1.0, np.nan, np.inf]
)
def test_classify_distance_rejects_invalid_thresholds(
    stop_distance,
    slow_distance
):
    with pytest.raises(DepthValidationError):
        classify_distance(2.0, stop_distance, slow_distance)

def test_median_depth_inside_bbox():
    depth = np.full((6, 6), 8.0, dtype=np.float32)
    depth[2:4, 2:4] = 2.5

    result = median_depth_in_bbox(
        depth,
        center_x=3.0,
        center_y=3.0,
        size_x=2.0,
        size_y=2.0,
        minimum_valid_pixels=4,
    )

    assert result == pytest.approx(2.5)

def test_median_depth_ignores_invalid_pixels():
    depth = np.array(
        [
            [np.nan, 0.0, 2.0],
            [np.inf, 4.0, 6.0]
        ],
        dtype=np.float32
    )

    result = median_depth_in_bbox(
        depth,
        center_x=1.5,
        center_y=1.0,
        size_x=3.0,
        size_y=2.0,
        minimum_valid_pixels=3
    )

    assert result == pytest.approx(4.0)

def test_median_depth_clips_bbox_to_image():
    depth = np.full((4, 4), 5.0, dtype=np.float32)

    result = median_depth_in_bbox(
        depth,
        center_x=0.0,
        center_y=0.0,
        size_x=4.0,
        size_y=4.0,
        minimum_valid_pixels=4
    )

    assert result == pytest.approx(5.0)

def test_median_depth_rejects_insufficient_valid_pixels():
    depth = np.full((4, 4), np.nan, dtype=np.float32)
    depth[0, 0] = 2.0

    with pytest.raises(DepthValidationError):
        median_depth_in_bbox(
            depth,
            center_x=2.0,
            center_y=2.0,
            size_x=4.0,
            size_y=4.0,
            minimum_valid_pixels=2,
        )

def test_median_depth_rejects_bbox_outside_image():
    depth = np.full((4, 4), 2.0, dtype=np.float32)

    with pytest.raises(DepthValidationError):
        median_depth_in_bbox(
            depth,
            center_x=20.0,
            center_y=20.0,
            size_x=2.0,
            size_y=2.0
        )

def test_median_depth_rejects_non_2d_image():
    depth = np.zeros((4, 4, 1), dtype=np.float132)

    with pytest.raises(DepthValidationError):
        median_depth_in_bbox(
            depth,
            center_x=2.0,
            center_y=2.0,
            size_x=2.0,
            size_y=2.0
        )