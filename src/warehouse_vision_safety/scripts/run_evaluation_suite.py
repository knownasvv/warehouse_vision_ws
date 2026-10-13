#!/usr/bin/env python3
"""Record six replay scenarios against an already running ROS pipeline."""

import argparse
from contextlib import suppress
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from build_evidence_report import build_report, scenario_assessment

REPLAY_NODE = '/rgbd_replay_publisher'
PARAMETERS = ('person_distance_m', 'invalid_depth_every_n', 'drop_depth_every_n')
SCENARIOS = (
    ('01_slow_2_5m', 'SLOW', (2.5, 0, 0)),
    ('02_stop_0_8m', 'STOP', (0.8, 0, 0)),
    ('03_clear_4_0m', 'CLEAR', (4.0, 0, 0)),
    ('04_invalid_depth', 'SENSOR_FAULT', (2.5, 1, 0)),
    ('05_missing_depth', 'SENSOR_FAULT', (2.5, 0, 1)),
    ('06_recovery_2_5m', 'SLOW', (2.5, 0, 0)),
)


def command(arguments, log, timeout=30):
    """Run a bounded command and stop its children on timeout or interruption."""
    log.write(f'\n$ {arguments!r}\n')
    log.flush()
    with subprocess.Popen(
        arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGINT)
            try:
                output, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
            log.write(output)
            log.flush()
            raise
    log.write(output)
    log.flush()
    if process.returncode:
        raise RuntimeError(f'Command failed ({process.returncode}): {arguments!r}\n{output}')
    return output.strip()


def configure(values, log):
    for name, value in zip(PARAMETERS, values):
        output = command(['ros2', 'param', 'set', REPLAY_NODE, name, str(value)], log)
        # Jazzy can return exit code zero even when a parameter is rejected.
        if 'Set parameter successful' not in output:
            raise RuntimeError(f'Parameter {name} was not accepted: {output}')


def wait_for_state(expected, log):
    output = command(
        [
            'ros2',
            'topic',
            'echo',
            '/safety/state',
            'std_msgs/msg/String',
            '--once',
            '--filter',
            f'm.data == {expected!r}',
        ],
        log,
    )
    if f'data: {expected}' not in output:
        raise RuntimeError(f'Did not observe state {expected}: {output}')
    print(f'Observed {expected}', flush=True)


def check_pipeline(log):
    nodes = command(['ros2', 'node', 'list'], log).splitlines()
    for name in (REPLAY_NODE, '/ai_person_detector', '/risk_monitor'):
        if nodes.count(name) != 1:
            raise RuntimeError(f'Expected exactly one running {name}; start ai_replay.launch.py')
    if '/scenario_publisher' in nodes or '/mock_person_detector' in nodes:
        raise RuntimeError('Stop the mock/scenario pipeline before recording AI replay')


def record_scenarios(duration, root, log):
    check_pipeline(log)
    originals = tuple(
        command(
            [
                'ros2',
                'param',
                'get',
                REPLAY_NODE,
                name,
                '--hide-type',
            ],
            log,
        )
        for name in PARAMETERS
    )
    # Validate the responses before changing any parameter.
    if not math.isfinite(float(originals[0])):
        raise RuntimeError('Replay distance is not finite')
    int(originals[1])
    int(originals[2])
    summaries = []
    try:
        for name, expected, values in SCENARIOS:
            # Missing depth must trigger a NEW fault after working depth delivery.
            if name == '05_missing_depth':
                configure((2.5, 0, 0), log)
                wait_for_state('SLOW', log)
            print(f'\n{name}: recording {duration:g} seconds after {expected}', flush=True)
            configure(values, log)
            wait_for_state(expected, log)
            output_directory = root / name
            command(
                [
                    'ros2',
                    'run',
                    'warehouse_vision_safety',
                    'evaluation_recorder',
                    '--ros-args',
                    '-p',
                    f'duration_s:={duration}',
                    '-p',
                    f'scenario_name:={name}',
                    '-p',
                    f'expected_state:={expected}',
                    '-p',
                    f'output_directory:={output_directory}',
                ],
                log,
                timeout=duration + 30,
            )
            summary = json.loads((output_directory / 'summary.json').read_text())
            if (
                summary.get('status') != 'completed'
                or summary.get('scenario_name') != name
                or summary.get('expected_state') != expected
            ):
                raise RuntimeError(f'{name}: missing or mismatched completed recording')
            summaries.append(summary)
    finally:
        # Attempt every restoration even if one parameter cannot be restored.
        restore_errors = []
        for name, value in zip(PARAMETERS, originals):
            try:
                output = command(['ros2', 'param', 'set', REPLAY_NODE, name, value], log)
                if 'Set parameter successful' not in output:
                    raise RuntimeError(output)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                restore_errors.append(f'{name}: {error}')
        if restore_errors:
            raise RuntimeError('Could not restore replay parameters: ' + '; '.join(restore_errors))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('duration', nargs='?', type=float, default=30.0)
    parser.add_argument('output_directory', nargs='?', type=Path)
    parser.add_argument('--test-result', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('duration must be finite and greater than zero')
    root = args.output_directory or Path('results') / datetime.now().strftime(
        'evaluation_suite_%Y%m%d_%H%M%S'
    )
    root = root.resolve()
    # Never replace an earlier evaluation, including an interrupted one.
    root.mkdir(parents=True, exist_ok=False)
    print(f'Results: {root}', flush=True)
    test_text = args.test_result.read_text() if args.test_result else ''
    if args.test_result:
        (root / 'colcon_test_result.txt').write_text(test_text)
    with (root / 'suite.log').open('w', buffering=1) as log:
        summaries = record_scenarios(args.duration, root, log)
    report_path = root / 'EVIDENCE_REPORT.md'
    report_path.write_text(build_report(summaries, test_text))
    print(f'Report: {report_path}', flush=True)
    if any(scenario_assessment(summary) != 'PASS' for summary in summaries):
        raise RuntimeError('One or more scenarios need REVIEW; see the report')
    print('All six scenarios passed', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f'Evaluation failed: {error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('Evaluation interrupted; partial results were preserved', file=sys.stderr)
        sys.exit(130)
