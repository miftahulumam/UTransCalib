from dotwiz import DotWiz

config_UTransCalib_resnet_oct4 = {
    'model_name': 'UTransCalib_resnet_oct4',
    'activation': 'nn.ReLU(inplace=True)',
    'init_weights': True
}

config_UTransCalib_lite_oct4 = {
    'model_name': 'UTransCalib_lite_oct4',
    'activation': 'nn.SiLU(inplace=True)',
    'init_weights': True
}

config_UTransCalib_densenet_oct20 = {
    'model_name': 'UTransCalib_densenet_oct20',
    'activation': 'nn.SiLU(inplace=True)',
    'init_weights': True
}

config_UTransCalib_mobilenet_oct20 = {
    'model_name': 'UTransCalib_mobilenet_oct20',
    'activation': 'nn.SiLU(inplace=True)',
    'depth_activation': 'nn.SiLU(inplace=True)',
    'rgb_activation': 'nn.SiLU(inplace=True)',
    'init_weights': False
}