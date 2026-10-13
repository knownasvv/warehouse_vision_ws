"""Exercise orchestration without requiring a ROS installation."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def suite(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        'evaluation_suite', scripts / 'run_evaluation_suite.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def simulated_cli(suite, monkeypatch):
    calls = []

    def run(args, log, timeout=30):
        calls.append(args)
        if args[1:3] == ['node', 'list']:
            return '/rgbd_replay_publisher\n/ai_person_detector\n/risk_monitor'
        if args[1:3] == ['param', 'get']:
            return '4.0' if args[4] == 'person_distance_m' else '0'
        if args[1:3] == ['param', 'set']:
            return 'Set parameter successful'
        if args[1:3] == ['topic', 'echo']:
            return 'data: ' + args[-1].split("'")[1]
        if args[1] == 'run':
            parameters = dict(value.split(':=', 1) for value in args if ':=' in value)
            output = Path(parameters['output_directory'])
            output.mkdir()
            (output / 'summary.json').write_text(
                json.dumps(
                    {
                        'status': 'completed',
                        'scenario_name': parameters['scenario_name'],
                        'expected_state': parameters['expected_state'],
                    }
                )
            )
            return ''
        raise AssertionError(args)

    monkeypatch.setattr(suite, 'command', run)
    return calls, run


def test_six_scenarios_recover_without_restarting_nodes(suite, simulated_cli, tmp_path):
    calls, _ = simulated_cli
    summaries = suite.record_scenarios(30.0, tmp_path, io.StringIO())
    assert [entry['scenario_name'] for entry in summaries] == [row[0] for row in suite.SCENARIOS]
    states = [args[-1] for args in calls if args[1:3] == ['topic', 'echo']]
    assert states == [
        f'm.data == {state!r}'
        for state in (
            'SLOW',
            'STOP',
            'CLEAR',
            'SENSOR_FAULT',
            'SLOW',
            'SENSOR_FAULT',
            'SLOW',
        )
    ]
    assert len([args for args in calls if args[1] == 'run']) == 6
    assert not any('launch' in args for args in calls)
    assert [args[-2:] for args in calls[-3:]] == [
        ['person_distance_m', '4.0'],
        ['invalid_depth_every_n', '0'],
        ['drop_depth_every_n', '0'],
    ]


def test_recorder_timeout_restores_original_parameters(
    suite, simulated_cli, monkeypatch, tmp_path
):
    calls, run = simulated_cli

    def timeout_recording(args, log, timeout=30):
        if args[1] == 'run':
            raise subprocess.TimeoutExpired(args, timeout)
        return run(args, log, timeout)

    monkeypatch.setattr(suite, 'command', timeout_recording)
    with pytest.raises(subprocess.TimeoutExpired):
        suite.record_scenarios(30.0, tmp_path, io.StringIO())
    assert [args[-2:] for args in calls[-3:]] == [
        ['person_distance_m', '4.0'],
        ['invalid_depth_every_n', '0'],
        ['drop_depth_every_n', '0'],
    ]


def test_parameter_rejection_with_zero_exit_status_is_an_error(suite, monkeypatch):
    monkeypatch.setattr(suite, 'command', lambda *args: 'Setting parameter failed')
    with pytest.raises(RuntimeError, match='not accepted'):
        suite.configure((2.5, 0, 0), io.StringIO())


def test_existing_results_are_never_replaced(suite, monkeypatch, tmp_path):
    evidence = tmp_path / 'summary.json'
    evidence.write_text('previous evidence')
    monkeypatch.setattr(sys, 'argv', ['suite', '30', str(tmp_path)])
    with pytest.raises(FileExistsError):
        suite.main()
    assert evidence.read_text() == 'previous evidence'


def test_command_timeout_stops_process(suite):
    log = io.StringIO()
    with pytest.raises(subprocess.TimeoutExpired):
        suite.command([sys.executable, '-c', 'import time; time.sleep(60)'], log, timeout=0.1)
