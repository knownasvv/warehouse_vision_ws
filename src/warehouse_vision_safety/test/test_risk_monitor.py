'''Regression checks requiring the ROS 2 runtime and message packages.'''

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('cv_bridge')

from rclpy.duration import Duration  # noqa: E402 (ROS availability checked above)
from rclpy.qos import ReliabilityPolicy  # noqa: E402 (ROS availability checked above)
from vision_msgs.msg import Detection2D, Detection2DArray  # noqa: E402 (ROS availability checked above)
from warehouse_vision_safety.risk_monitor import RiskMonitor  # noqa: E402 (ROS availability checked above)
from warehouse_vision_safety.scenario_publisher import ScenarioPublisher  # noqa: E402 (ROS availability checked above)


@pytest.fixture
def monitor():
    rclpy.init()
    node = RiskMonitor()
    node.state_publisher = Mock()
    node.distance_publisher = Mock()
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def feed_frame(monitor, distance, has_person=True):
    depth = np.full((10, 10), distance, dtype=np.float32)
    message = monitor.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
    message.header.stamp = monitor.get_clock().now().to_msg()
    detections = Detection2DArray()
    detections.header = message.header
    if has_person:
        detection = Detection2D()
        detection.bbox.center.position.x = 5.0
        detection.bbox.center.position.y = 5.0
        detection.bbox.size_x = 10.0
        detection.bbox.size_y = 10.0
        detections.detections.append(detection)
    monitor.synchronized_callback(message, detections)


@pytest.mark.parametrize(
    'distance,state', [(0.8, 'STOP'), (2.5, 'SLOW'), (4.0, 'CLEAR')]
)
def test_valid_frame_publishes_one_finite_distance(monitor, distance, state):
    feed_frame(monitor, distance)
    monitor.state_publisher.publish.assert_called_once()
    monitor.distance_publisher.publish.assert_called_once()
    assert monitor.state_publisher.publish.call_args.args[0].data == state
    assert monitor.distance_publisher.publish.call_args.args[0].data == pytest.approx(
        distance
    )


def test_invalid_depth_publishes_fault_and_recovers(monitor):
    feed_frame(monitor, np.nan)
    assert monitor.state_publisher.publish.call_args.args[0].data == 'SENSOR_FAULT'
    assert math.isnan(monitor.distance_publisher.publish.call_args.args[0].data)
    feed_frame(monitor, 2.5)
    assert monitor.state_publisher.publish.call_args.args[0].data == 'SLOW'
    assert monitor.distance_publisher.publish.call_args.args[0].data == pytest.approx(
        2.5
    )
    assert monitor.distance_publisher.publish.call_count == 2


def test_no_person_publishes_clear_without_a_distance(monitor):
    feed_frame(monitor, 2.5, has_person=False)
    assert monitor.state_publisher.publish.call_args.args[0].data == 'CLEAR'
    assert math.isnan(monitor.distance_publisher.publish.call_args.args[0].data)


def test_missing_synchronized_input_remains_a_fault(monitor):
    monitor.last_synchronized_input = monitor.get_clock().now() - Duration(seconds=2)
    monitor.check_staleness()
    assert monitor.state_publisher.publish.call_args.args[0].data == 'SENSOR_FAULT'
    assert math.isnan(monitor.distance_publisher.publish.call_args.args[0].data)


def test_depth_conversion_failure_is_a_fault(monitor):
    monitor.bridge = SimpleNamespace(
        imgmsg_to_cv2=Mock(side_effect=ValueError('bad depth'))
    )
    monitor.synchronized_callback(None, Detection2DArray())
    assert monitor.state_publisher.publish.call_args.args[0].data == 'SENSOR_FAULT'


def test_simulated_depth_endpoints_request_reliable_delivery(monitor):
    publisher = ScenarioPublisher()
    try:
        assert (
            monitor.depth_subscriber.sub.qos_profile.reliability
            == ReliabilityPolicy.RELIABLE
        )
        assert (
            publisher.depth_publisher.qos_profile.reliability
            == ReliabilityPolicy.RELIABLE
        )
    finally:
        publisher.destroy_node()
