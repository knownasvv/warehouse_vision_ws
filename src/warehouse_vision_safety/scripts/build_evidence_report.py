#!/usr/bin/env python3

import argparse
from datetime import datetime
from datetime import timezone
import json
import os
from pathlib import Path
import platform
import subprocess

EXPECTED_DISTANCES = {
    'ai_replay_slow_2_5m': 2.5,
    '01_slow_2_5m': 2.5,
    '02_stop_0_8m': 0.8,
    '03_clear_4_0m': 4.0,
    '06_recovery_2_5m': 2.5,
}


def format_number(value, decimals=2, suffix=''):
    if value is None:
        return 'N/A'

    return f'{value:.{decimals}f}{suffix}'


def git_revision():
    result = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        return result.stdout.strip()

    return 'unavailable'


def memory_total_gib():
    meminfo = Path('/proc/meminfo')

    if not meminfo.exists():
        return 'unavailable'

    for line in meminfo.read_text(encoding='utf-8').splitlines():
        if line.startswith('MemTotal:'):
            kibibytes = int(line.split()[1])
            return f'{kibibytes / 1024 / 1024:.2f} GiB'

    return 'unavailable'


def opencv_version():
    try:
        import cv2

        return cv2.__version__
    except ImportError:
        return 'unavailable'


def load_summaries(results_root):
    summaries = []

    for summary_path in sorted(results_root.glob('*/summary.json')):
        with summary_path.open(encoding='utf-8') as file:
            summaries.append(json.load(file))

    return summaries


def scenario_assessment(summary):
    name = summary['scenario_name']
    detection_rate = summary['detector']['detection_rate_percent']
    state_agreement = summary['safety']['state_agreement_percent']
    mean_distance = summary['depth']['distance_m']['mean']

    checks = [
        summary.get('status') == 'completed',
        detection_rate >= 95.0,
        state_agreement >= 95.0,
    ]

    expected_distance = EXPECTED_DISTANCES.get(name)

    if expected_distance is not None:
        checks.append(
            mean_distance is not None
            and abs(mean_distance - expected_distance) <= 0.10
        )

    if name in ('04_invalid_depth', '05_missing_depth'):
        checks.extend([
            summary['expected_state'] == 'SENSOR_FAULT',
            summary['depth']['distance_m']['count'] == 0,
            summary['depth']['invalid_distance_messages'] > 0,
        ])

    return 'PASS' if all(checks) else 'REVIEW'


def build_report(summaries, test_result_text):
    generated_at = datetime.now(timezone.utc).isoformat()
    backends = sorted({
        summary['detector'].get('backend', 'unavailable')
        for summary in summaries
    })

    lines = [
        '# Project Evidence Report',
        '',
        f'Generated: `{generated_at}`',
        f'Git revision: `{git_revision()}`',
        '',
        '## Environment',
        '',
        '| Property | Value |',
        '|---|---|',
        f"| ROS distribution | {os.environ.get('ROS_DISTRO', 'unavailable')} |",
        f'| Operating system | {platform.platform()} |',
        f'| Architecture | {platform.machine()} |',
        f'| Kernel | {platform.release()} |',
        f'| Memory | {memory_total_gib()} |',
        f'| Python | {platform.python_version()} |',
        f'| OpenCV | {opencv_version()} |',
        f"| Recorded inference backend | {', '.join(backends)} |",
        '| Model | YOLO11n ONNX |',
        '',
        '## Evaluation results',
        '',
        '| Scenario | Detect % | State % | Mean depth | '
        'Mean latency | P95 latency | Assessment |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]

    for summary in summaries:
        detector = summary['detector']
        safety = summary['safety']
        depth = summary['depth']['distance_m']
        latency = summary['performance']['inference_time_ms']

        lines.append(
            f"| {summary['scenario_name']} "
            f"| {format_number(detector['detection_rate_percent'], 1, '%')} "
            f"| {format_number(safety['state_agreement_percent'], 1, '%')} "
            f"| {format_number(depth['mean'], 3, ' m')} "
            f"| {format_number(latency['mean'], 1, ' ms')} "
            f"| {format_number(latency['p95'], 1, ' ms')} "
            f'| {scenario_assessment(summary)} |'
        )

    lines.extend(
        [
            '',
            '## Scenario details',
            '',
        ]
    )

    for summary in summaries:
        state_counts = summary['safety']['state_counts']
        state_text = ', '.join(
            f'`{state}`: {count}'
            for state, count in sorted(state_counts.items())
        )

        lines.extend(
            [
                f"### {summary['scenario_name']}",
                '',
                f"- Expected state: `{summary['expected_state']}`",
                f"- Recording status: `{summary.get('status', 'unavailable')}`",
                (
                    '- Recorded inference backend: '
                    f"{summary['detector'].get('backend', 'unavailable')}"
                ),
                f"- State observations: {state_text or 'None'}",
                (
                    '- Person detection messages: '
                    f"{summary['detector']['detection_messages']}"
                ),
                (
                    '- Frames containing a detected person: '
                    f"{summary['detector']['frames_with_person']}"
                ),
                (
                    '- Valid distance messages: '
                    f"{summary['depth']['distance_m']['count']}"
                ),
                (
                    '- Invalid distance messages: '
                    f"{summary['depth']['invalid_distance_messages']}"
                ),
                '',
            ]
        )

    lines.extend(
        [
            '## Acceptance criteria',
            '',
            '- Recording must have completed.',
            '- Detection rate: at least 95%.',
            '- Expected safety-state agreement: at least 95%.',
            '- Invalid/missing-depth trials must report SENSOR_FAULT, '
            'no valid distances, and at least one invalid distance.',
            '- Normal-scenario mean distance: within 0.10 m of '
            'the configured synthetic distance.',
            '- Inference latency is reported rather than hidden; '
            'P95 below 500 ms indicates capacity for the configured '
            '2 Hz input rate.',
            '',
            '## Automated test result',
            '',
            '```text',
            test_result_text.strip() or 'Test output was not provided.',
            '```',
            '',
            '## Interpretation constraints',
            '',
            '- Results apply to the deterministic replay scenario.',
            (
                '- Distance statistics include only valid samples; compare the '
                'valid and invalid message counts before interpreting the mean.'
            ),
            (
                '- Backend labels come from recorded summaries; older recordings '
                'may contain incorrect labels and are not rewritten by this report.'
            ),
            '- The depth map is synthetic and manually aligned with the image.',
            '- Detection success on one replay image does not establish '
            'general real-world accuracy.',
            '- Timing results apply to this ARM64 virtual machine.',
            '- This system estimates risk state but does not control a robot.',
            '',
        ]
    )

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results-root',
        type=Path,
        default=Path('results/evaluation_suite'),
    )
    parser.add_argument(
        '--test-result',
        type=Path,
        default=Path('results/colcon_test_result.txt'),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/EVIDENCE_REPORT.md'),
    )
    arguments = parser.parse_args()

    summaries = load_summaries(arguments.results_root)

    if not summaries:
        raise SystemExit(
            f'No summary.json files found under {arguments.results_root}'
        )

    test_result_text = ''
    if arguments.test_result.exists():
        test_result_text = arguments.test_result.read_text(
            encoding='utf-8'
        )

    report = build_report(summaries, test_result_text)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(report, encoding='utf-8')

    print(f'Wrote {arguments.output}')
    print(f'Included {len(summaries)} evaluation scenarios')


if __name__ == '__main__':
    main()
