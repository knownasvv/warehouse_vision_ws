import cv2
from cv_bridge import CvBridge

import numpy as np

import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class ScenarioPublisher(Node):

    def __init__(self):
        super().__init__('scenario_publisher')

        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('person_distance_m', 2.5)
        self.declare_parameter('background_distance_m', 8.0)
        self.declare_parameter('invalid_depth_every_n', 0)
        self.declare_parameter('drop_depth_every_n', 0)

        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.frame_count = 0

        self.bridge = CvBridge()

        self.color_publisher = self.create_publisher(
            Image,
            '/sim_camera/color/image_raw',
            qos_profile_sensor_data,
        )
        self.depth_publisher = self.create_publisher(
            Image,
            '/sim_camera/depth/image_raw',
            qos_profile_sensor_data,
        )
        self.camera_info_publisher = self.create_publisher(
            CameraInfo,
            '/sim_camera/color/camera_info',
            qos_profile_sensor_data,
        )

        rate_hz = max(float(self.get_parameter('rate_hz').value), 0.1)
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_frame)

        self.get_logger().info(
            f'Publishing simulated RGB-D data at {rate_hz:.1f} Hz'
        )

    def create_rgb_image(self):
        image = np.full(
            (self.height, self.width, 3),
            (55, 55, 55),
            dtype=np.uint8,
        )

        center_x = self.width // 2
        head_center = (center_x, self.height // 3)
        head_radius = 35

        body_left = center_x - 55
        body_right = center_x + 55
        body_top = head_center[1] + head_radius
        body_bottom = body_top + 180

        cv2.circle(image, head_center, head_radius, (70, 180, 240), -1)
        cv2.rectangle(
            image,
            (body_left, body_top),
            (body_right, body_bottom),
            (70, 180, 240),
            -1,
        )

        distance = float(self.get_parameter('person_distance_m').value)
        cv2.putText(
            image,
            f'Simulated person: {distance:.2f} m',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        return image, head_center, head_radius, (
            body_left,
            body_top,
            body_right,
            body_bottom,
        )

    def create_depth_image(
        self,
        head_center,
        head_radius,
        body_bounds,
        invalid_depth,
    ):
        background_distance = float(
            self.get_parameter('background_distance_m').value
        )
        person_distance = float(
            self.get_parameter('person_distance_m').value
        )

        depth = np.full(
            (self.height, self.width),
            background_distance,
            dtype=np.float32,
        )

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(mask, head_center, head_radius, 255, -1)

        left, top, right, bottom = body_bounds
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)

        if invalid_depth:
            depth[mask > 0] = np.nan
        else:
            depth[mask > 0] = person_distance

        return depth

    def create_camera_info(self, stamp):
        message = CameraInfo()
        message.header.stamp = stamp
        message.header.frame_id = 'sim_camera_optical_frame'
        message.width = self.width
        message.height = self.height

        focal_length = 525.0
        center_x = self.width / 2.0
        center_y = self.height / 2.0

        message.k = [
            focal_length, 0.0, center_x,
            0.0, focal_length, center_y,
            0.0, 0.0, 1.0,
        ]
        message.p = [
            focal_length, 0.0, center_x, 0.0,
            0.0, focal_length, center_y, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        return message

    def publish_frame(self):
        self.frame_count += 1

        invalid_every = int(
            self.get_parameter('invalid_depth_every_n').value
        )
        drop_every = int(
            self.get_parameter('drop_depth_every_n').value
        )

        invalid_depth = (
            invalid_every > 0
            and self.frame_count % invalid_every == 0
        )
        drop_depth = (
            drop_every > 0
            and self.frame_count % drop_every == 0
        )

        rgb, head_center, head_radius, body_bounds = (
            self.create_rgb_image()
        )
        depth = self.create_depth_image(
            head_center,
            head_radius,
            body_bounds,
            invalid_depth,
        )

        stamp = self.get_clock().now().to_msg()

        color_message = self.bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
        color_message.header.stamp = stamp
        color_message.header.frame_id = 'sim_camera_optical_frame'

        self.color_publisher.publish(color_message)
        self.camera_info_publisher.publish(
            self.create_camera_info(stamp)
        )

        if not drop_depth:
            depth_message = self.bridge.cv2_to_imgmsg(
                depth,
                encoding='32FC1',
            )
            depth_message.header.stamp = stamp
            depth_message.header.frame_id = 'sim_camera_optical_frame'
            self.depth_publisher.publish(depth_message)


def main(args=None):
    rclpy.init(args=args)
    node = ScenarioPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
