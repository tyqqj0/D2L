# -*- CODING: UTF-8 -*-
# @time 2024/3/21 14:27
# @Author tyqqj
# @File runner.py
# @
# @Aim 


# from main import main
import subprocess

all = ['--amp',
       'run_name', '1',
       '--exp_name', 'expression_data',
       '--group_size', '0'
       ]

parameters = [
    # ['--model', 'resnet18', '--noise_ratio', '0'],
    ['--model', 'resnet50', '--noise_ratio', '0'],
    # ['--model', 'resnet18', '--noise_ratio', '0.5'],
    ['--model', 'resnet50', '--noise_ratio', '0.1'],
    ['--model', 'resnet50', '--noise_ratio', '0.2'],
    ['--model', 'resnet50', '--noise_ratio', '0.3'],
    ['--model', 'resnet50', '--noise_ratio', '0.4'],
    ['--model', 'resnet50', '--noise_ratio', '0.5'],
    ['--model', 'resnet50', '--noise_ratio', '0.6'],
    ['--model', 'resnet50', '--noise_ratio', '0.7'],
    ['--model', 'resnet50', '--noise_ratio', '0.8'],
    ['--model', 'resnet50', '--noise_ratio', '0.9'],
    ['--model', 'resnet50', '--noise_ratio', '1']
]

for parameter in parameters:
    print(parameter + all)
    command = ['python', 'main.py'] + parameter + all
    subprocess.run(command)
    print('Finish run', parameter)
