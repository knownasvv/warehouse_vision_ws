from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class RGBDReplayPublisher(Node):
    # Approximate person regions for the supplied bus.jpg image.
    # Coordinate are normalized: x_min, y_min, x_max, y_max
    PERSON_REGIONS = (
        (0.00, 0.50, 0.11, 0.87), # Partially visible left person
        (0.03, 0.35, 0.31, 0.86), # Main left person
        (0.24, 0.35, 0.46, 0.82), # Centre person
        (0.80, 0.34, 1.00, 0.88), # Right person
    )

    def __init__(self):
        super().__init__('rgbd_replay_publisher')

        default_image_path = (
            Path(get_package_share_directory('warehouse_vision_safety'))
            / 'test_data'
            / 'bus.jpg'
        )

        self.declare_parameter('image_path', str(default_image_path))
        self.declare_parameter('rate_hz', 2.0)
        self.declare_parameter('person_distance_m', 2.5)
        self.declare_parameter('background_distance_m', 8.0)
        self.declare_parameter('invalid_depth_every_n', 0)
        self.declare_parameter('drop_depth_every_n', 0)

        image_path = self.get_parameter('image_path').get_parameter_value().string_value
        rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        if rate_hz <= 0.0:
            raise ValueError('rate_hz must be greater than zero')

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f'Could not load replay image: {image_path}')

        self.height, self.width = self.image.shape[:2]
        self.bridge = CvBridge()
        self.frame_count = 0

        self.rgb_publisher = self.create_publisher(
            Image,
            '/sim_camera/color/image_raw',
            10,
        )

        self.depth_publisher = self.create_publisher(
            Image,
            '/sim_camera/depth/image_raw',
            10,
        )

        self.camera_info_publisher = self.create_publisher(
            CameraInfo,
            '/sim_camera/color/camera_info',
            10,
        )

        self.timer = self.create_timer(
            1.0 / rate_hz,
            self.publish_frame
        )

        self.get_logger().info(
            f'RGB-D replay ready: {image_path}, '
            f'{self.width}x{self.height}, {rate_hz:.1f} Hz'
        )

    def apply_person_regions(self, depth_image, value):
        for x_min_n, y_min_n, x_max_n, y_max_n in self.PERSON_REGIONS:
            x_min = int(round(x_min_n * self.width))
            y_min = int(round(y_min_n * self.height))
            x_max = int(round(x_max_n * self.width))
            y_max = int(round(y_max_n * self.height))

            depth_image[y_min:y_max, x_min:x_max] = value

    def create_camera_info(self, stamp):
        camera_info = CameraInfo()
        camera_info.header.stamp = stamp
        camera_info.header.frame_id = 'sim_camera_optical_frame'
        camera_info.width = self.width
        camera_info.height = self.height
        camera_info.distortion_model = 'plumb_bob'

        focal_length = 600.0
        center_x = self.width / 2.0
        center_y = self.height / 2.0

        camera_info.k = [
            focal_length, 0.0, center_x,
            0.0, focal_length, center_y,
            0.0, 0.0, 1.0,
        ]

        camera_info.p = [
            focal_length, 0.0, center_x, 0.0,
            0.0, focal_length, center_y, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        return camera_info

    def publish_frame(self):
        self.frame_count += 1

        person_distance = self.get_parameter('person_distance_m').get_parameter_value().double_value
        background_distance = self.get_parameter('background_distance_m').get_parameter_value().double_value
        invalid_every_n = self.get_parameter('invalid_depth_every_n').get_parameter_value().integer_value
        drop_every_n = self.get_parameter('drop_depth_every_n').get_parameter_value().integer_value

        if person_distance <= 0.0 or background_distance <= 0.0:
            self.get_logger().error(
                'Depth distances must be greater than zero',
                throttle_duration_sec=2.0,
            )
            return

        stamp = self.get_clock().now().to_msg()

        rgb_message = self.bridge.cv2_to_imgmsg(
            self.image,
            encoding='bgr8',
        )
        rgb_message.header.stamp = stamp
        rgb_message.header.frame_id = 'sim_camera_optical_frame'

        depth_image = np.full(
            (self.height, self.width),
            background_distance,
            dtype=np.float32,
        )
        self.apply_person_regions(depth_image, person_distance)

        if (
            invalid_every_n > 0
            and self.frame_count % invalid_every_n == 0
        ):
            depth_image.fill(np.nan)

        should_drop_depth = (
            drop_every_n > 0
            and self.frame_count % drop_every_n == 0
        )

        self.rgb_publisher.publish(rgb_message)
        self.camera_info_publisher.publish(
            self.create_camera_info(stamp)
        )

        if not should_drop_depth:
            depth_message = self.bridge.cv2_to_imgmsg(
                depth_image,
                encoding='32FC1',
            )
            depth_message.header.stamp = stamp
            depth_message.header.frame_id = 'sim_camera_optical_frame'
            self.depth_publisher.publish(depth_message)


def main(args=None):
    rclpy.init(args=args)
    node = RGBDReplayPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()