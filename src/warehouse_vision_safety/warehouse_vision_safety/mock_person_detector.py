import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray

class MockPersonDetector(Node):
    def __init__(self):
        super().__init__("mock_person_detector")

        self.declare_parameter("center_x", 320.0)
        self.declare_parameter("center_y", 250.0)
        self.declare_parameter("size_x", 110.0)
        self.declare_parameter("size_y", 250.0)

        self.publisher = self.create_publisher(
            Detection2DArray,
            "/perception/person_detections",
            qos_profile_sensor_data,
        )

        self.subscription = self.create_subscription(
            Image,
            "/sim_camera/color/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info("Mock person detector started")        

    def image_callback(self, image_message):
        detection = Detection2D()
        detection.header = image_message.header
        detection.id = "simulated_person_0"

        detection.bbox.center.position.x = float(self.get_parameter("center_x").value)
        detection.bbox.center.position.y = float(self.get_parameter("center_y").value)
        detection.bbox.size_x = float(self.get_parameter("size_x").value)
        detection.bbox.size_y = float(self.get_parameter("size_y").value)

        message = Detection2DArray()
        message.header = image_message.header
        message.detections.append(detection)

        self.publisher.publish(message)

def main(args=None):
    rclpy.init(args=args)
    node = MockPersonDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()