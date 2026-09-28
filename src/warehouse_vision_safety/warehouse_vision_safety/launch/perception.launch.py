import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    package_directory = get_package_share_directory("warehouse_vision_safety")

    default_config = os.path.join(package_directory, "config", "safety.yaml")
    config_argument = DeclareLaunchArgument(
        "config",
        default_value=default_config,
        description="Path to the perception configuration file"
    )
    config = LaunchConfiguration("config")

    scenario_publisher = Node(
        package="warehouse_vision_safety",
        executable="scenario_publisher",
        name="scenario_publisher",
        output="screen",
        parameters=[config]
    )

    mock_detector = Node(
        package="warehouse_vision_safety",
        executable="mock_person_detector",
        name="mock_person_detector",
        output="screen",
        parameters=[config]
    )

    risk_monitor = Node(
        package="warehouse_vision_safety",
        executable="risk_monitor",
        name="risk_monitor",
        output="screen",
        parameters=[config]
    )

    return LaunchDescription(
        [
            config_argument,
            scenario_publisher,
            mock_detector,
            risk_monitor
        ]
    )