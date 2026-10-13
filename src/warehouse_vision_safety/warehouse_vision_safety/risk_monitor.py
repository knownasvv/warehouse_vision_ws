import math
from time import monotonic

from cv_bridge import CvBridge

import message_filters
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection2DArray

from warehouse_vision_safety.safety_logic import (
    classify_distance,
    DepthValidationError,
    median_depth_in_bbox
)


class RiskMonitor(Node):

    def __init__(self):
        super().__init__('risk_monitor')

        self.declare_parameter('stop_distance_m', 1.0)
        self.declare_parameter('slow_distance_m', 3.0)
        self.declare_parameter('stale_timeout_s', 1.0)
        self.declare_parameter('minimum_valid_pixels', 20)
        self.declare_parameter('diagnostics_enabled', False)

        self.bridge = CvBridge()
        self.last_synchronized_input = self.get_clock().now()
        self.last_state = None
        self._last_sync_monotonic = monotonic()
        self._last_timing_report = monotonic()
        self._input_timing = {}

        self.state_publisher = self.create_publisher(
            String,
            '/safety/state',
            10,
        )
        self.distance_publisher = self.create_publisher(
            Float32,
            '/perception/person_distance_m',
            10,
        )

        self.depth_subscriber = message_filters.Subscriber(
            self,
            Image,
            '/sim_camera/depth/image_raw',
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )

        self.detection_subscriber = message_filters.Subscriber(
            self,
            Detection2DArray,
            '/perception/person_detections',
            qos_profile=qos_profile_sensor_data
        )

        # Observe the existing subscriptions without creating extra image traffic.
        self.depth_subscriber.registerCallback(self._track_input, 'depth')
        self.detection_subscriber.registerCallback(self._track_input, 'detections')

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.depth_subscriber, self.detection_subscriber],
            queue_size=10,
            slop=0.1,
        )
        self.synchronizer.registerCallback(self.synchronized_callback)

        self.watchdog_timer = self.create_timer(0.2, self.check_staleness)

        self.get_logger().info('Risk monitor started')

    def publish_result(self, state, distance=math.nan, reason=''):
        state_message = String()
        state_message.data = state
        self.state_publisher.publish(state_message)

        distance_message = Float32()
        distance_message.data = float(distance)
        self.distance_publisher.publish(distance_message)

        if state != self.last_state:
            detail = f', reason{reason}' if reason else ''
            distance_text = f'{distance:.2f} m' if math.isfinite(distance) else 'unavailable'
            self.get_logger().info(
                f'State changed to {state}, distance={distance_text}{detail}'
            )
            self.last_state = state

    def synchronized_callback(
        self,
        depth_message,
        detections_message,
    ):
        self.last_synchronized_input = self.get_clock().now()
        self._last_sync_monotonic = monotonic()
        self._track_input(depth_message, 'pairs')

        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_message,
                desired_encoding='32FC1'
            )
        except Exception as error:
            self.get_logger().error(f'Depth conversion failed: {error}')
            self.publish_result('SENSOR_FAULT', reason='depth conversion failed')
            return

        if not detections_message.detections:
            self.publish_result('CLEAR', reason='no person detected')
            return

        distances = []
        for detection in detections_message.detections:
            try:
                distance = median_depth_in_bbox(
                    depth_image=depth_image,
                    center_x=detection.bbox.center.position.x,
                    center_y=detection.bbox.center.position.y,
                    size_x=detection.bbox.size_x,
                    size_y=detection.bbox.size_y,
                    minimum_valid_pixels=self.get_parameter('minimum_valid_pixels').value
                )
            except DepthValidationError as error:
                self.get_logger().warning(f'Cannot estimate person distance: {error}')
                self.publish_result('SENSOR_FAULT')
                return

            distances.append(distance)

        if not distances:
            self.publish_result('CLEAR')
            return

        nearest_distance = min(distances)

        try:
            state = classify_distance(
                distance_m=nearest_distance,
                stop_distance_m=self.get_parameter('stop_distance_m').value,
                slow_distance_m=self.get_parameter('slow_distance_m').value
            )
        except DepthValidationError as error:
            self.get_logger().error(f'Invalid safety configuration: {error}')
            self.publish_result('SENSOR_FAULT')
            return

        self.publish_result(state, distance=nearest_distance)

    def _track_input(self, message, name):
        if not self.get_parameter('diagnostics_enabled').value or message is None:
            return
        now = monotonic()
        previous = self._input_timing.get(name)
        gap_ms = 0.0 if previous is None else (now - previous['arrival']) * 1000.0
        stamp = message.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        age_ms = (self.get_clock().now().nanoseconds - stamp_ns) / 1_000_000.0
        previous = previous or {}
        self._input_timing[name] = {
            'arrival': now,
            'stamp_ns': stamp_ns,
            'count': previous.get('count', 0) + 1,
            'max_gap_ms': max(previous.get('max_gap_ms', 0.0), gap_ms),
            'max_age_ms': max(previous.get('max_age_ms', age_ms), age_ms),
        }

    def _log_input_timing(self, ros_elapsed):
        if not self.get_parameter('diagnostics_enabled').value:
            self._input_timing.clear()
            self._last_timing_report = monotonic()
            return
        now = monotonic()
        if now - self._last_timing_report < 1.0:
            return
        window = now - self._last_timing_report
        self._last_timing_report = now
        parts = []
        for name in ('depth', 'detections', 'pairs'):
            sample = self._input_timing.get(name)
            if sample is None:
                parts.append(f'{name}=none')
                continue
            parts.append(
                f"{name}[n={sample['count']} "
                f"idle_ms={(now - sample['arrival']) * 1000.0:.1f} "
                f"max_gap_ms={sample['max_gap_ms']:.1f} "
                f"max_age_ms={sample.get('max_age_ms', math.nan):.1f}]"
            )
            sample['count'] = 0
            sample['max_gap_ms'] = 0.0
            sample.pop('max_age_ms', None)
        depth = self._input_timing.get('depth')
        detections = self._input_timing.get('detections')
        stamp_delta = math.nan
        if depth is not None and detections is not None:
            stamp_delta = (depth['stamp_ns'] - detections['stamp_ns']) / 1_000_000.0
        self.get_logger().info(
            f'[DEBUG-risk-timing] window_s={window:.2f} '
            f'sync_idle_ros_ms={ros_elapsed * 1000.0:.1f} '
            f'sync_idle_mono_ms={(now - self._last_sync_monotonic) * 1000.0:.1f} '
            f'latest_stamp_delta_ms={stamp_delta:.1f} ' + ' '.join(parts)
        )

    def check_staleness(self):
        timeout = float(self.get_parameter('stale_timeout_s').value)
        elapsed = (self.get_clock().now() -
                   self.last_synchronized_input).nanoseconds / 1_000_000_000.0

        if elapsed > timeout:
            self.publish_result('SENSOR_FAULT', reason='synchronized input is stale')

        self._log_input_timing(elapsed)


def main(args=None):
    rclpy.init(args=args)
    node = RiskMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
