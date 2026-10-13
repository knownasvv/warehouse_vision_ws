import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable


def generate_launch_description():
    package_share = get_package_share_directory('warehouse_vision_safety')

    config_path = os.path.join(package_share, 'config', 'safety.yaml')

    return LaunchDescription(
        [
            Node(
                package='warehouse_vision_safety',
                executable='rgbd_replay_publisher',
                name='rgbd_replay_publisher',
                parameters=[config_path],
                output='screen'
            ),
            Node(
                package="warehouse_vision_safety",
                executable="ai_person_detector",
                name="ai_person_detector",
                parameters=[config_path],
                output="screen",
                prefix=[EnvironmentVariable('WAREHOUSE_VISION_PYTHON')]
            ),
            Node(
                package="warehouse_vision_safety",
                executable="risk_monitor",
                name="risk_monitor",
                parameters=[config_path],
                output="screen",
            ),
        ]
    )