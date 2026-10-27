import sys
if sys.prefix == '/home/lola/Documentos/ROS2/.pixi/envs/default':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/lola/Documentos/ROS2/supergiz_ws/install/ia'
