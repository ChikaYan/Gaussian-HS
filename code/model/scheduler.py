import numpy as np


class Schedule():
    def __init__(self):
        self.total_step = 0
        pass

    def __call__(self, step:int):
        return 0

class ConstantSchedule(Schedule):
    def __init__(self, init_value:float, end_value:float=None, total_step:int=None):
        self.init_value = init_value
        self.total_step = total_step
    
    def __call__(self, step:int):
        return self.init_value

class LinearSchedule(Schedule):
    def __init__(self, init_value:float, end_value:float, total_step:int):
        self.init_value = init_value
        self.end_value = end_value
        self.total_step = total_step
    
    def __call__(self, step:int):
        if step < self.total_step:
            return (self.end_value - self.init_value) * step / self.total_step + self.init_value
        else:
            return self.end_value

class ExpSchedule(Schedule):
    def __init__(self, init_value:float, end_value:float, total_step:int):
        self.init_value = init_value
        self.end_value = end_value
        self.total_step = total_step
    
    def __call__(self, step:int):
        if step < self.total_step:
            t = np.clip(step / self.total_step, 0, 1)
            return np.exp(np.log(self.init_value) * (1 - t) + np.log(self.end_value) * t)
        else:
            return self.end_value

class SequentialSchedule(Schedule):
    def __init__(self, schedules:list[Schedule]):
        self.schedules = schedules
    
    def __call__(self, step:int):
        for i, s in enumerate(self.schedules):                
            if step < s.total_step:
                return s(step)
            elif i == len(self.schedules) - 1:
                return s(step)
            else:
                step -= s.total_step

