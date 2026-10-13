# Fault-Aware Human Proximity Estimation with ROS 2

A simulation-only ROS 2 perception project that combines pretrained person detection with aligned depth data to estimate human proximity and produce fail-safe safety states.

The system uses YOLO11n through OpenCV DNN for RGB person detection, extracts the median valid depth inside each detected bounding box, and publishes one of four states:

- `CLEAR`
- `SLOW`
- `STOP`
- `SENSOR_FAULT`

## Why this project exists

Robots operating near people require more than object detection. They must combine perception sources, handle invalid sensor data, and expose failures explicitly instead of silently making unsafe decisions.

This project explores that problem through a small, reproducible ROS 2 pipeline running entirely in simulation on a CPU-only ARM64 virtual machine.

## Architecture

```mermaid
flowchart LR
    A[RGB-D replay publisher] -->|RGB image| B[YOLO11n person detector]
    A -->|32FC1 depth image| C[Risk monitor]
    B -->|Detection2DArray| C
    B -->|Inference latency| E[Evaluation recorder]
    C -->|Nearest distance| E
    C -->|Safety state| E
    A -. Invalid or missing depth .-> C
    C --> D{CLEAR / SLOW / STOP / SENSOR_FAULT}
```

## Main components

| Node | Responsibility |
|---|---|
| `rgbd_replay_publisher` | Publishes a repeatable RGB image and aligned synthetic depth map |
| `ai_person_detector` | Runs YOLO11n ONNX inference with OpenCV DNN |
| `risk_monitor` | Fuses detections and depth and determines the safety state |
| `evaluation_recorder` | Records detection, distance, state and latency statistics |
| `scenario_publisher` | Publishes the earlier geometric test scenario |
| `mock_person_detector` | Provides deterministic bounding boxes for initial integration tests |

## Safety policy

Default thresholds are:

| Condition | State |
|---|---|
| Distance ≤ 1.0 m | `STOP` |
| 1.0 m < distance ≤ 3.0 m | `SLOW` |
| Distance > 3.0 m | `CLEAR` |
| Invalid or stale synchronized depth | `SENSOR_FAULT` |

The boundary comparisons are inclusive. For example, exactly `1.0 m` produces `STOP`.

## Published topics

| Topic | Type | Description |
|---|---|---|
| `/sim_camera/color/image_raw` | `sensor_msgs/Image` | Simulated RGB input |
| `/sim_camera/depth/image_raw` | `sensor_msgs/Image` | Aligned 32-bit floating-point depth |
| `/perception/person_detections` | `vision_msgs/Detection2DArray` | Detected person boxes |
| `/perception/person_distance_m` | `std_msgs/Float32` | Nearest estimated person distance |
| `/perception/inference_time_ms` | `std_msgs/Float32` | Per-frame inference time |
| `/perception/debug_image` | `sensor_msgs/Image` | RGB image with detection overlays |
| `/safety/state` | `std_msgs/String` | Current safety state |

## Platform

The project was developed using:

- ROS 2 Jazzy
- Ubuntu 24.04 LTS
- ARM64 UTM/QEMU virtual machine
- 6 virtual CPU cores
- 8 GB memory
- CPU-only OpenCV DNN inference
- No physical robot or camera hardware

## Quick start

The following assumes ROS 2 Jazzy is already installed.

### Install dependencies

```bash
sudo apt update
sudo apt install -y \
  python3-opencv \
  python3-pytest \
  wget \
  ros-jazzy-cv-bridge \
  ros-jazzy-vision-msgs \
  ros-jazzy-message-filters
```

### Download the model and replay image

```bash
cd src/warehouse_vision_safety
mkdir -p models test_data

wget -O models/yolo11n.onnx \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx

wget -O test_data/bus.jpg \
  https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg

echo "634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b  models/yolo11n.onnx" \
  | sha256sum --check

cd ../..
```

### Build

```bash
source /opt/ros/jazzy/setup.bash

colcon build \
  --packages-select warehouse_vision_safety \
  --symlink-install

source install/setup.bash
```

### Launch the AI pipeline

```bash
ros2 launch warehouse_vision_safety ai_replay.launch.py
```

With the default simulated person distance of `2.5 m`, the expected state is `SLOW`.

View the outputs with:

```bash
ros2 topic echo /safety/state
ros2 topic echo /perception/person_distance_m
ros2 topic echo /perception/inference_time_ms
```

The annotated image is available at:

```text
/perception/debug_image
```

## Testing

Run the unit tests:

```bash
colcon test \
  --packages-select warehouse_vision_safety \
  --event-handlers console_direct+

colcon test-result --verbose
```

The unit tests cover:

- Safety threshold boundaries
- Invalid and non-finite distances
- Invalid threshold configurations
- Median depth extraction
- Bounding-box clipping
- Insufficient valid depth pixels
- Bounding boxes outside the image
- Invalid depth-image dimensions

## Evaluation

Start the AI replay pipeline, then run:

```bash
./src/warehouse_vision_safety/scripts/run_evaluation_suite.sh \
  15.0 \
  results/evaluation_suite
```

The suite evaluates:

1. `SLOW` at 2.5 m
2. `STOP` at 0.8 m
3. `CLEAR` at 4.0 m
4. Corrupted all-NaN depth
5. Missing depth messages
6. Recovery after a sensor fault

### Measured results

These values must be replaced with the actual generated measurements.

| Scenario | Detection rate | State agreement | Mean distance | Mean inference | P95 inference |
|---|---:|---:|---:|---:|---:|
| SLOW, 2.5 m | PENDING | PENDING | PENDING | PENDING | PENDING |
| STOP, 0.8 m | PENDING | PENDING | PENDING | PENDING | PENDING |
| CLEAR, 4.0 m | PENDING | PENDING | PENDING | PENDING | PENDING |
| Invalid depth | PENDING | PENDING | N/A | PENDING | PENDING |
| Missing depth | PENDING | PENDING | N/A | PENDING | PENDING |
| Recovery, 2.5 m | PENDING | PENDING | PENDING | PENDING | PENDING |

Raw summaries and latency samples are stored under:

```text
results/evaluation_suite/
```

## Design decisions

### ONNX with OpenCV DNN

YOLO11n is executed as an ONNX model through OpenCV DNN. This avoids a large PyTorch runtime and is suitable for the CPU-only ARM64 virtual machine.

### Median depth rather than one pixel

The risk monitor uses the median of valid depth pixels inside each detected person box. A single center pixel can fall on the background, clothing edge or an invalid depth location. The median is more resistant to isolated outliers.

### Explicit fault state

Missing or invalid depth does not produce `CLEAR`. It produces `SENSOR_FAULT`, preventing sensor failure from being interpreted as a safe environment.

### Separate perception and safety nodes

Detection and safety decisions are separate ROS nodes. The detector can therefore be replaced without rewriting the depth-fusion and safety logic.

## Limitations

- The RGB-D replay is controlled synthetic input, not a physical RGB-D measurement.
- Foreground depth regions are manually aligned with the replay photograph.
- The system uses a pretrained model and does not train a new neural network.
- Bounding boxes are processed independently; there is no temporal tracking.
- The project estimates proximity but does not command robot motion.
- The camera calibration is approximate and is not used for full 3D localization.
- Performance measurements reflect one ARM64 virtual-machine configuration.
- A detection failure is not equivalent to proof that no person is present.

## Future work

- Replace replay data with a rendered Gazebo RGB-D camera.
- Add tracked detections across frames.
- Publish standard ROS diagnostic messages.
- Estimate 3D person positions using camera intrinsics.
- Add rosbag-based regression tests.
- Connect safety states to a simulated mobile-robot controller.

## Third-party assets

- YOLO11n model: Ultralytics model assets
- Test photograph: Ultralytics assets repository

The model and image remain subject to their respective upstream licensing terms.
