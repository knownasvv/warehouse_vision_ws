from glob import glob

import os

from setuptools import find_packages, setup


package_name = 'warehouse_vision_safety'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml*')),
        (os.path.join('share', package_name, 'models'), glob('models/*.onnx')),
        (os.path.join('share', package_name, 'test_data'), glob('test_data/*.jpg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vaemini',
    maintainer_email='72638950+knownasvv@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'scenario_publisher = warehouse_vision_safety.scenario_publisher:main',
            'mock_person_detector = warehouse_vision_safety.mock_person_detector:main',
            'risk_monitor = warehouse_vision_safety.risk_monitor:main',
            'ai_person_detector = warehouse_vision_safety.ai_person_detector:main',
            'rgbd_replay_publisher = warehouse_vision_safety.rgbd_replay_publisher:main',
            'evaluation_recorder = warehouse_vision_safety.evaluation_recorder:main',
        ],
    },
)
