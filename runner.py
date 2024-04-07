# -*- CODING: UTF-8 -*-
# @time 2024/3/21 14:27
# @Author tyqqj
# @File runner.py
# @
# @Aim 


# from main import main
import subprocess

all = ['--amp',
       '--run_name', 'show_ne',
       # '--exp_name', 'noisy_test'
       # 'run_name', '1',
       '--exp_name', 'n8',
       '--group_size', '200',
       '--model', 'resnet50'
       ]

parameters = [
    ['--noise_ratio', '0.00'],
    ['--noise_ratio', '0.05'],
    ['--noise_ratio', '0.1'],
    ['--noise_ratio', '0.15'],
    ['--noise_ratio', '0.2'],
    ['--noise_ratio', '0.25'],
    ['--noise_ratio', '0.3'],
    ['--noise_ratio', '0.35'],
    ['--noise_ratio', '0.4'],
    ['--noise_ratio', '0.45'],
    ['--noise_ratio', '0.5'],
    ['--noise_ratio', '0.6'],
    ['--noise_ratio', '0.7'],
    ['--noise_ratio', '0.8'],
    ['--noise_ratio', '0.9'],
    ['--noise_ratio', '1']
]
for i in range(3):
    for parameter in parameters:
        print(parameter + all)
        command = ['python', 'main.py'] + parameter + all
        subprocess.run(command)
        print('Finish run', parameter)
