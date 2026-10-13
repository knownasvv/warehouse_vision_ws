import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32, String
from vision_msgs.msg import Detection2DArray


def numeric_summary(values):
    if not values:
        return {
            'count': 0,
            'minimum': None,
            'mean': None,
            'median': None,
            'p95': None,
            'maximum': None,
            'standard_deviation': None,
        }

    array = np.asarray(values, dtype=np.float64)

    return {
        'count': int(array.size),
        'minimum': float(np.min(array)),
        'mean': float(np.mean(array)),
        'median': float(np.median(array)),
        'p95': float(np.percentile(array, 95)),
        'maximum': float(np.max(array)),
        'standard_deviation': float(np.std(array)),
    }


class EvaluationRecorder(Node):
    '''Record quantitative measurements from an AI perception trial.'''

    def __init__(self):
        super().__init__('evaluation_recorder')

        self.declare_parameter('duration_s', 30.0)
        self.declare_parameter('scenario_name', 'ai_replay_slow_2_5m')
        self.declare_parameter('expected_state', 'SLOW')
        self.declare_parameter(
            'output_directory',
            'results/ai_replay_evaluation',
        )

        self.duration_s = self.get_parameter(
            'duration_s'
        ).get_parameter_value().double_value

        self.scenario_name = self.get_parameter(
            'scenario_name'
        ).get_parameter_value().string_value

        self.expected_state = self.get_parameter(
            'expected_state'
        ).get_parameter_value().string_value

        output_directory = self.get_parameter(
            'output_directory'
        ).get_parameter_value().string_value

        if self.duration_s <= 0.0:
            raise ValueError('duration_s must be greater than zero')

        self.output_directory = Path(
            output_directory
        ).expanduser().resolve()

        self.start_time = perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.results_written = False

        self.detection_message_count = 0
        self.frames_with_person = 0
        self.total_person_detections = 0

        self.confidence_samples = []
        self.inference_samples_ms = []
        self.distance_samples_m = []
        self.invalid_distance_count = 0
        self.state_counts = Counter()

        self.create_subscription(
            Detection2DArray,
            '/perception/person_detections',
            self.detection_callback,
            10,
        )

        self.create_subscription(
            Float32,
            '/perception/inference_time_ms',
            self.inference_callback,
            10,
        )

        self.create_subscription(
            Float32,
            '/perception/person_distance_m',
            self.distance_callback,
            10,
        )

        self.create_subscription(
            String,
            '/safety/state',
            self.state_callback,
            10,
        )

        self.timer = self.create_timer(0.25, self.check_duration)

        self.get_logger().info(
            f'Recording scenario \'{self.scenario_name}\' for '
            f'{self.duration_s:.1f} seconds'
        )

    def detection_callback(self, message):
        self.detection_message_count += 1

        if message.detections:
            self.frames_with_person += 1

        self.total_person_detections += len(message.detections)

        for detection in message.detections:
            for result in detection.results:
                if result.hypothesis.class_id == 'person':
                    self.confidence_samples.append(
                        float(result.hypothesis.score)
                    )

    def inference_callback(self, message):
        value = float(message.data)

        if np.isfinite(value) and value >= 0.0:
            self.inference_samples_ms.append(value)

    def distance_callback(self, message):
        value = float(message.data)

        if np.isfinite(value) and value > 0.0:
            self.distance_samples_m.append(value)
        else:
            self.invalid_distance_count += 1

    def state_callback(self, message):
        self.state_counts[message.data] += 1

    def check_duration(self):
        if perf_counter() - self.start_time < self.duration_s:
            return

        self.write_results('completed')
        rclpy.shutdown()

    def write_results(self, status):
        if self.results_written:
            return

        elapsed_s = perf_counter() - self.start_time
        self.output_directory.mkdir(parents=True, exist_ok=True)

        detection_rate = 0.0
        if self.detection_message_count:
            detection_rate = (
                self.frames_with_person
                / self.detection_message_count
                * 100.0
            )

        total_state_samples = sum(self.state_counts.values())
        expected_state_samples = self.state_counts.get(
            self.expected_state,
            0,
        )

        state_agreement = 0.0
        if total_state_samples:
            state_agreement = (
                expected_state_samples
                / total_state_samples
                * 100.0
            )

        inference_summary = numeric_summary(
            self.inference_samples_ms
        )

        mean_inference_ms = inference_summary['mean']
        estimated_max_fps = None

        if mean_inference_ms is not None and mean_inference_ms > 0.0:
            estimated_max_fps = 1000.0 / mean_inference_ms

        summary = {
            'status': status,
            'scenario_name': self.scenario_name,
            'expected_state': self.expected_state,
            'started_at_utc': self.started_at,
            'elapsed_s': elapsed_s,
            'detector': {
                'model': 'YOLO11n ONNX',
                'backend': 'OpenCV DNN CPU',
                'detection_messages': self.detection_message_count,
                'frames_with_person': self.frames_with_person,
                'detection_rate_percent': detection_rate,
                'total_person_detections': (
                    self.total_person_detections
                ),
                'confidence': numeric_summary(
                    self.confidence_samples
                ),
            },
            'performance': {
                'inference_time_ms': inference_summary,
                'estimated_max_fps': estimated_max_fps,
            },
            'depth': {
                'distance_m': numeric_summary(
                    self.distance_samples_m
                ),
                'invalid_distance_messages': (
                    self.invalid_distance_count
                ),
            },
            'safety': {
                'state_counts': dict(self.state_counts),
                'state_agreement_percent': state_agreement,
            },
        }

        summary_path = self.output_directory / 'summary.json'
        with summary_path.open('w', encoding='utf-8') as output_file:
            json.dump(summary, output_file, indent=2)

        latency_path = (
            self.output_directory / 'inference_latency.csv'
        )
        with latency_path.open(
            'w',
            encoding='utf-8',
            newline='',
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(['sample_index', 'inference_time_ms'])

            for index, value in enumerate(
                self.inference_samples_ms,
                start=1,
            ):
                writer.writerow([index, value])

        self.results_written = True

        self.get_logger().info(
            f'Evaluation complete: {summary_path}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = EvaluationRecorder()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if not node.results_written:
                node.write_results('interrupted')
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
