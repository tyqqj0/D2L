# -*- CODING: UTF-8 -*-
# @time 2024/3/21 14:27
# @Author tyqqj
# @File runner.py
# @
# @Aim 


# from main import main
import subprocess

parameters = [
    ['--model', 'resnet18', '--noise_ratio', '0']
    , ['--model', 'resnet50', '--noise_ratio', '0.5']
    , ['--model', 'resnet18', '--noise_ratio', '0.5']
    , ['--model', 'resnet50', '--noise_ratio', '0']
]

for parameter in parameters:
    print(parameter)
    command = ['python', 'main.py'] + parameter
    subprocess.run(command)
    print('Finish run', parameter)
