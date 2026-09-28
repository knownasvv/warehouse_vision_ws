import math

import message_filters
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection2DArray

class RiskMonitor(Node):
    def __init__(self):
        super().__init__("risk_monitor")

        self.declare_parameter("stop_distance_m", 1.0)
        self.declare_parameter("slow_distance_m", 3.0)
        self.declare_parameter("stale_timeout_s", 1.0)
        self.declare_parameter("minimum_valid_pixels", 20)

        self.bridge = CvBridge()
        self.last_synchronized_input = self.get_clock().now()
        self.last_state = None

        self.state_publisher = self.create_publisher(
            String,
            "/safety/state",
            10,
        )
        self.distance_publisher = self.create_publisher(
            Float32,
            "/perception/person_distance_m",
            10,
        )

        self.depth_subscriber = message_filters.Subscriber(
            self,
            Image,
            "/sim_camera/depth/image_raw",
            qos_profile=qos_profile_sensor_data
        )

        self.detection_subscriber = message_filters.Subscriber(
            self,
            Detection2DArray,
            "/perception/person_detections",
            qos_profile=qos_profile_sensor_data
        )

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.depth_subscriber, self.detection_subscriber],
            queue_size=10,
            slop=0.1,
        )
        self.synchronizer.registerCallback(self.synchronized_callback)

        self.watchdog_timer = self.create_timer(0.2, self.check_staleness)

        self.get_logger().info("Risk monitor started")

    def publish_result(self, state, distance=math.nan, reason=""):
        state_message = String()
        state_message.data = state
        self.state_publisher.publish(state_message)

        distance_message = Float32()
        distance_message.data = float(distance)
        self.distance_publisher.publish(distance_message)

        if state != self.last_state:
            detail = f", reason(reason)" if reason else ""
            distance_text = f"{distance:.2f} m" if math.isfinite(distance) else "unavailable"
            self.get_logger().info(
                f"State changed to {state}, distance={distance_text}{detail}"
            )
            self.last_state = state

    def synchronized_callback(
        self,
        depth_message,
        detections_message,
    ):
        self.last_synchronized_input = self.get_clock().now()

        try:
            depth = self.bridge.imgmsg_to_cv2(
                depth_message,
                desired_encoding="32FC1"
            )
        except Exception as error:
            self.get_logger().error(f"Depth conversion failed: {error}")
            self.publish_result("SENSOR_FAULT", reason="depth conversion failed")
            return

        if not detections_message.detections:
            self.publish_result("CLEAR", reason="no person detected")
            return

        distances = []
        for detection in detections_message.detections:
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            size_x = detection.bbox.size_x
            size_y = detection.bbox.size_y

            x_min = max(0, int(math.floor(center_x - size_x / 2.0)))
            x_max = min(depth.shape[1], int(math.ceil(center_x + size_x / 2.0)))
            y_min = max(0, int(math.floor(center_y - size_y / 2.0)))
            y_max = min(depth.shape[0], int(math.ceil(center_y + size_y / 2.0)))

            if x_min >= x_max or y_min >= y_max:
                self.publish_result("SENSOR_FAULT", reason="invalid detection bounds")
                return

            region = depth[y_min:y_max, x_min:x_max]
            valid = region[np.isfinite(region) & (region > 0.0)]

            minimum_valid_pixels = int(self.get_parameter("minimum_valid_pixels").value)
            if valid.size < minimum_valid_pixels:
                self.publish_result("SENSOR_FAULT", reason="insufficient valid depth")
                return

            distances.append(float(np.median(valid)))

        nearest_distance = min(distances)
        stop_distance = float(self.get_parameter("stop_distance_m").value)
        slow_distance = float(self.get_parameter("slow_distance_m").value)

        if nearest_distance <= stop_distance:
            state = "STOP"
        elif nearest_distance <= slow_distance:
            state = "SLOW"
        else:
            state = "CLEAR"
        self.publish_result(state, nearest_distance)

    def check_staleness(self):
        timeout = float(self.get_parameter("stale_timeout_s").value)
        elapsed = (self.get_clock().now() - self.last_synchronized_input).nanoseconds / 1_000_000_000.0

        if elapsed > timeout:
            self.publish_result("SENSOR_FAULT", reason="synchronized input is stale")    

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

if __name__ == "__main__":
    main()