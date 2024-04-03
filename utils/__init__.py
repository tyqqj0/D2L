# -*- CODING: UTF-8 -*-
# @time 2024/1/25 12:00
# @Author tyqqj
# @File __init__.py.py
# @
# @Aim

import time

class EpochTimer:
    def __init__(self):
        self.start = None
        self.end = None

    def _start(self):
        self.start = time.time()
        self.end = None

    def _end(self):
        if self.start is None:
            raise ValueError('Timer not started')
        if self.end is None:
            self.end = time.time()
        t = self.end - self.start
        speed_pm = t / 60
        speed_ph = speed_pm / 60
        return t, speed_pm, speed_ph

    def __str__(self):
        return f'Using {self._end()[0]}, speed: {self._end()[1]:.2f}m, {self._end()[2]:.2f}h'

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'{self.name} took {time.time() - self.start:.2f} seconds')
        return False