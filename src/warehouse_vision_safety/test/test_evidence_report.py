'''Check reporting against recorded data, including failed historical trials.'''

import importlib.util
from pathlib import Path

import pytest

REPORT_PATH = (
    Path(__file__).resolve().parents[1] / 'scripts' / 'build_evidence_report.py'
)
spec = importlib.util.spec_from_file_location('build_evidence_report', REPORT_PATH)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def summary():
    return {
        'status': 'completed',
        'scenario_name': 'ai_replay_slow_2_5m',
        'expected_state': 'SLOW',
        'detector': {
            'backend': 'ONNX Runtime CPU',
            'detection_rate_percent': 100.0,
            'detection_messages': 60,
            'frames_with_person': 60,
        },
        'safety': {'state_agreement_percent': 100.0, 'state_counts': {'SLOW': 60}},
        'depth': {
            'distance_m': {'mean': 2.5, 'count': 60},
            'invalid_distance_messages': 0,
        },
        'performance': {'inference_time_ms': {'mean': 47.2, 'p95': 60.4}},
    }


def test_failed_run_stays_review_and_exposes_sparse_depth(summary):
    summary['safety'] = {
        'state_agreement_percent': 2 / 142 * 100,
        'state_counts': {'SENSOR_FAULT': 140, 'SLOW': 2},
    }
    summary['depth']['distance_m']['count'] = 2
    summary['depth']['invalid_distance_messages'] = 142
    summary['detector']['backend'] = 'OpenCV DNN CPU'
    text = report.build_report([summary], '')
    assert report.scenario_assessment(summary) == 'REVIEW'
    assert 'Valid distance messages: 2' in text
    assert 'Invalid distance messages: 142' in text
    assert '| Recorded inference backend | OpenCV DNN CPU |' in text


def test_successful_completed_trial_passes(summary):
    assert report.scenario_assessment(summary) == 'PASS'
    assert 'ONNX Runtime CPU' in report.build_report([summary], '')


@pytest.mark.parametrize('distance', [None, 4.0])
def test_named_slow_trial_checks_depth(summary, distance):
    summary['depth']['distance_m']['mean'] = distance
    assert report.scenario_assessment(summary) == 'REVIEW'


def test_interrupted_trial_cannot_pass(summary):
    summary['status'] = 'interrupted'
    assert report.scenario_assessment(summary) == 'REVIEW'


def test_mixed_recordings_retain_backend_labels(summary):
    from copy import deepcopy

    old = deepcopy(summary)
    old['detector']['backend'] = 'OpenCV DNN CPU'
    text = report.build_report([old, summary], '')
    assert '| Recorded inference backend | ONNX Runtime CPU, OpenCV DNN CPU |' in text
