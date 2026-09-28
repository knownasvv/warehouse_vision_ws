import math

import numpy as np


class DepthValidationError(ValueError):
    """Raised when a reliable distance cannot be extracted."""


def classify_distance(
    distance_m: float,
    stop_distance_m: float,
    slow_distance_m: float,
) -> str:
    if stop_distance_m <= 0.0:
        raise DepthValidationError('stop_distance_m must be greater than zero')

    if slow_distance_m <= stop_distance_m:
        raise DepthValidationError(
            'slow_distance_m must be greater than stop_distance_m'
        )

    if not math.isfinite(distance_m) or distance_m <= 0.0:
        raise DepthValidationError('distance must be finite and greater than zero')

    if distance_m <= stop_distance_m:
        return 'STOP'

    if distance_m <= slow_distance_m:
        return 'SLOW'

    return 'CLEAR'


def median_depth_in_bbox(
    depth_image: np.ndarray,
    center_x: float,
    center_y: float,
    size_x: float,
    size_y: float,
    minimum_valid_pixels: int = 20,
) -> float:
    if depth_image.ndim != 2:
        raise DepthValidationError('depth image must be two-dimensional')

    if minimum_valid_pixels <= 0:
        raise DepthValidationError('minimum_valid_pixels must be greater than zero')

    bbox_values = (center_x, center_y, size_x, size_y)

    if not all(math.isfinite(value) for value in bbox_values):
        raise DepthValidationError('bounding-box values must be finite')

    if size_x <= 0.0 or size_y <= 0.0:
        raise DepthValidationError('bounding-box dimensions must be positive')

    image_height, image_width = depth_image.shape

    x_min = max(0, math.floor(center_x - size_x / 2.0))
    x_max = min(image_width, math.ceil(center_x + size_x / 2.0))
    y_min = max(0, math.floor(center_y - size_y / 2.0))
    y_max = min(image_height, math.ceil(center_y + size_y / 2.0))

    if x_min >= x_max or y_min >= y_max:
        raise DepthValidationError('bounding box does not overlap the depth image')

    depth_region = depth_image[y_min:y_max, x_min:x_max]
    valid_depths = depth_region[
        np.isfinite(depth_region) & (depth_region > 0.0)
    ]

    if valid_depths.size < minimum_valid_pixels:
        raise DepthValidationError(
            f'only {valid_depths.size} valid depth pixels were found; '
            f'{minimum_valid_pixels} are required'
        )

    return float(np.median(valid_depths))
