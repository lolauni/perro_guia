"""Percepción del perro guía: percepcion_fusion con los parámetros de config/percepcion.yaml.

  ros2 launch ia percepcion.launch.py                    # con ventana de demo
  ros2 launch ia percepcion.launch.py ventana:=false     # sin pantalla
  ros2 launch ia percepcion.launch.py params:=/ruta/a/otro.yaml
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    por_defecto = os.path.join(get_package_share_directory('ia'), 'config', 'percepcion.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=por_defecto),
        DeclareLaunchArgument('ventana', default_value='true'),
        Node(
            package='ia',
            executable='percepcion_fusion',
            name='percepcion_completa',
            output='screen',
            parameters=[LaunchConfiguration('params'),
                        {'debug_view': ParameterValue(LaunchConfiguration('ventana'), value_type=bool)}],
        ),
    ])
