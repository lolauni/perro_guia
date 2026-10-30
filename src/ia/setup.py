from glob import glob

from setuptools import find_packages, setup

package_name = 'ia'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lola',
    maintainer_email='lola@todo.com',
    description='Paquete de IA para ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_yolo = ia.detector_yolo:main',
            'publicador_foto = ia.publicador_foto:main',
            'segmentador_suelo = ia.segmentador_suelo:main',
            'percepcion_completa = ia.percepcion_completa:main',
            'percepcion_3d = ia.percepcion_3d:main',
            'percepcion_fusion = ia.percepcion_fusion:main',
        ],
    },
)