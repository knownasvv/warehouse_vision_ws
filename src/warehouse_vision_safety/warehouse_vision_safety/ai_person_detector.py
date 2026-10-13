from pathlib import Path
from time import perf_counter

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class AIPersonDetector(Node):

    def __init__(self):
        super().__init__('ai_person_detector')

        default_model_path = (
            Path(get_package_share_directory('warehouse_vision_safety'))
            / 'models'
            / 'yolo11n.onnx'
        )

        self.declare_parameter('model_path', str(default_model_path))
        self.declare_parameter('confidence_threshold', 0.40)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('publish_debug_image', True)

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = (
            self.get_parameter('confidence_threshold').get_parameter_value().double_value
        )
        self.nms_threshold = self.get_parameter('nms_threshold').get_parameter_value().double_value
        self.input_size = self.get_parameter('input_size').get_parameter_value().integer_value
        self.publish_debug_image = (
            self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        )

        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f'YOLO model was not found at: {self.model_path}')

        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError('confidence_threshold must be between 0 and 1')

        if not 0.0 < self.nms_threshold <= 1.0:
            raise ValueError('nms_threshold must be between 0 and 1')

        if self.input_size <= 0:
            raise ValueError('input_size must be greater than zero')

        self.bridge = CvBridge()

        self.session = ort.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider'],
        )
        self.input_name = self.session.get_inputs()[0].name

        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            '/perception/person_detections',
            10
        )

        self.inference_time_publisher = self.create_publisher(
            Float32,
            '/perception/inference_time_ms',
            10
        )

        self.debug_image_publisher = self.create_publisher(
            Image,
            '/perception/debug_image',
            10
        )

        self.image_subscription = self.create_subscription(
            Image,
            '/sim_camera/color/image_raw',
            self.image_callback,
            QoSProfile(
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )

        self.get_logger().info(
            f'Loaded YOLO11n from {self.model_path}, '
            f'confidence={self.confidence_threshold:.2f}, '
            f'NMS={self.nms_threshold:.2f}, '
            f'input={self.input_size}x{self.input_size}'
        )

    def letterbox(self, image):
        image_height, image_width = image.shape[:2]

        scale = min(
            self.input_size / image_width,
            self.input_size / image_height
        )

        resized_width = int(round(image_width * scale))
        resized_height = int(round(image_height * scale))

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR
        )

        padded = np.full(
            (self.input_size, self.input_size, 3),
            114,
            dtype=np.uint8
        )

        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2

        padded[
            pad_y:pad_y + resized_height,
            pad_x:pad_x + resized_width,
        ] = resized

        return padded, scale, pad_x, pad_y

    def run_inference(self, image):
        prepared_image, scale, pad_x, pad_y = self.letterbox(image)

        blob = cv2.dnn.blobFromImage(
            prepared_image,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )

        raw_output = self.session.run(
            None,
            {self.input_name: blob},
        )[0]

        predictions = np.squeeze(raw_output)

        if predictions.ndim != 2:
            raise RuntimeError(f'Unexpected YOLO output shape: {raw_output.shape}')

        # Standard YOLO11 ONNX output: (84, 8400)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        if predictions.shape[1] < 5:
            raise RuntimeError(f'Unexpected YOLO prediction shape: {predictions.shape}')

        image_height, image_width = image.shape[:2]
        boxes = []
        scores = []

        for prediction in predictions:
            # COCO class zero is "person"
            person_score = float(prediction[4])

            if person_score < self.confidence_threshold:
                continue

            center_x, center_y, box_width, box_height = prediction[:4]

            x_min = (center_x - box_width / 2.0 - pad_x) / scale
            y_min = (center_y - box_height / 2.0 - pad_y) / scale
            x_max = (center_x + box_width / 2.0 - pad_x) / scale
            y_max = (center_y + box_height / 2.0 - pad_y) / scale

            x_min = max(0.0, min(float(image_width), float(x_min)))
            y_min = max(0.0, min(float(image_height), float(y_min)))
            x_max = max(0.0, min(float(image_width), float(x_max)))
            y_max = max(0.0, min(float(image_height), float(y_max)))

            clipped_width = x_max - x_min
            clipped_height = y_max - y_min

            if clipped_width <= 1.0 or clipped_height <= 1.0:
                continue

            boxes.append(
                [
                    int(round(x_min)),
                    int(round(y_min)),
                    int(round(clipped_width)),
                    int(round(clipped_height))
                ]
            )
            scores.append(person_score)

        if not boxes:
            return []

        selected_indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence_threshold,
            self.nms_threshold
        )

        if len(selected_indices) == 0:
            return []

        detections = []

        for index in np.asarray(selected_indices).reshape(-1):
            x, y, width, height = boxes[int(index)]
            score = scores[int(index)]

            detections.append((x, y, width, height, score))

        return detections

    def create_detection_message(self, source_message, detections):
        output = Detection2DArray()
        output.header = source_message.header

        for index, (x, y, width, height, score) in enumerate(detections):
            detection = Detection2D()
            detection.header = source_message.header
            detection.id = f'person_{index}'

            detection.bbox.center.position.x = x + width / 2.0
            detection.bbox.center.position.y = y + height / 2.0
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = float(width)
            detection.bbox.size_y = float(height)

            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = 'person'
            result.hypothesis.score = float(score)
            detection.results.append(result)

            output.detections.append(detection)

        return output

    def publish_debug_view(self, source_message, image, detections):
        if not self.publish_debug_image:
            return

        annotated = image.copy()

        for x, y, width, height, score in detections:
            cv2.rectangle(
                annotated,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f'person {score:.2f}',
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        debug_message = self.bridge.cv2_to_imgmsg(
            annotated,
            encoding='bgr8'
        )
        debug_message.header = source_message.header
        self.debug_image_publisher.publish(debug_message)

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding='bgr8'
            )

            start_time = perf_counter()
            detections = self.run_inference(image)
            inference_time_ms = (perf_counter() - start_time) * 1000.0

            detection_message = self.create_detection_message(
                message,
                detections
            )
            self.detection_publisher.publish(detection_message)

            timing_message = Float32()
            timing_message.data = float(inference_time_ms)
            self.inference_time_publisher.publish(timing_message)

            self.publish_debug_view(message, image, detections)

        except Exception as error:
            # Do not publish an empty detection message on processing failure.
            # The risk monitor's stale-input timer can then report SENSOR_FAULT.
            self.get_logger().error(
                f'Person detection failed: {error}',
                throttle_duration_sec=2.0
            )

def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = AIPersonDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()