# Project Evidence Report

Generated: `2026-10-14T04:04:59.932490+00:00`
Git revision: `fd40fef`

## Environment

| Property | Value |
|---|---|
| ROS distribution | jazzy |
| Operating system | Linux-7.0.0-38-generic-aarch64-with-glibc2.39 |
| Architecture | aarch64 |
| Kernel | 7.0.0-38-generic |
| Memory | 7.73 GiB |
| Python | 3.12.3 |
| OpenCV | 4.6.0 |
| Recorded inference backend | ONNX Runtime CPU |
| Model | YOLO11n ONNX |

## Evaluation results

| Scenario | Detect % | State % | Mean depth | Mean latency | P95 latency | Assessment |
|---|---:|---:|---:|---:|---:|---|
| 01_slow_2_5m | 100.0% | 100.0% | 2.500 m | 38.4 ms | 54.0 ms | PASS |
| 02_stop_0_8m | 100.0% | 100.0% | 0.800 m | 37.8 ms | 49.9 ms | PASS |
| 03_clear_4_0m | 100.0% | 100.0% | 4.000 m | 37.3 ms | 52.8 ms | PASS |
| 04_invalid_depth | 100.0% | 100.0% | N/A | 36.6 ms | 52.3 ms | PASS |
| 05_missing_depth | 100.0% | 100.0% | N/A | 40.3 ms | 52.2 ms | PASS |
| 06_recovery_2_5m | 100.0% | 100.0% | 2.500 m | 37.4 ms | 49.3 ms | PASS |

## Scenario details

### 01_slow_2_5m

- Expected state: `SLOW`
- Recording status: `completed`
- Recorded inference backend: ONNX Runtime CPU
- State observations: `SLOW`: 61
- Person detection messages: 58
- Frames containing a detected person: 58
- Valid distance messages: 61
- Invalid distance messages: 0

### 02_stop_0_8m

- Expected state: `STOP`
- Recording status: `completed`
- Recorded inference backend: ONNX Runtime CPU
- State observations: `STOP`: 61
- Person detection messages: 61
- Frames containing a detected person: 61
- Valid distance messages: 61
- Invalid distance messages: 0

### 03_clear_4_0m

- Expected state: `CLEAR`
- Recording status: `completed`
- Recorded inference backend: ONNX Runtime CPU
- State observations: `CLEAR`: 60
- Person detection messages: 60
- Frames containing a detected person: 60
- Valid distance messages: 60
- Invalid distance messages: 0

### 04_invalid_depth

- Expected state: `SENSOR_FAULT`
- Recording status: `completed`
- Recorded inference backend: ONNX Runtime CPU
- State observations: `SENSOR_FAULT`: 60
- Person detection messages: 60
- Frames containing a detected person: 60
- Valid distance messages: 0
- Invalid distance messages: 60

### 05_missing_depth

- Expected state: `SENSOR_FAULT`
- Recording status: `completed`
- Recorded inference backend: ONNX Runtime CPU
- State observations: `SENSOR_FAULT`: 150
- Person detection messages: 60
- Frames containing a detected person: 60
- Valid distance messages: 0
- Invalid distance messages: 150

### 06_recovery_2_5m

- Expected state: `SLOW`
- Recording status: `completed`
- Recorded inference backend: ONNX Runtime CPU
- State observations: `SLOW`: 60
- Person detection messages: 60
- Frames containing a detected person: 60
- Valid distance messages: 60
- Invalid distance messages: 0

## Acceptance criteria

- Recording must have completed.
- Detection rate: at least 95%.
- Expected safety-state agreement: at least 95%.
- Invalid/missing-depth trials must report SENSOR_FAULT, no valid distances, and at least one invalid distance.
- Normal-scenario mean distance: within 0.10 m of the configured synthetic distance.
- Inference latency is reported rather than hidden; P95 below 500 ms indicates capacity for the configured 2 Hz input rate.

## Automated test result

```text
Summary: 49 tests, 0 errors, 0 failures, 1 skipped
```

## Interpretation constraints

- Results apply to the deterministic replay scenario.
- Distance statistics include only valid samples; compare the valid and invalid message counts before interpreting the mean.
- Backend labels come from recorded summaries; older recordings may contain incorrect labels and are not rewritten by this report.
- The depth map is synthetic and manually aligned with the image.
- Detection success on one replay image does not establish general real-world accuracy.
- Timing results apply to this ARM64 virtual machine.
- This system estimates risk state but does not control a robot.
