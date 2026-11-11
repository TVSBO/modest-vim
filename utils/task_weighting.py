"""Bounded two-task DWA (MTAN, CVPR 2019), using scalar epoch monitors.

CE+Dice excludes the signed topology term. Bounds are a project safeguard.
Reference: https://github.com/lorenmt/mtan
"""
import math


class TaskWeighting:
    def __init__(self, mode='fixed', temperature=2.0, minimum=0.5):
        if mode not in ('fixed', 'dwa'):
            raise ValueError('task_weighting must be fixed or dwa')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('DWA temperature must be finite and positive')
        if not math.isfinite(minimum) or not 0 < minimum <= 1:
            raise ValueError('DWA minimum must be in (0, 1]')
        self.mode = mode
        self.temperature = float(temperature)
        self.minimum = float(minimum)
        self.history = []

    def weights(self):
        if self.mode == 'fixed' or len(self.history) < 2:
            return {'depth': 1.0, 'seg': 1.0}
        earlier, later = self.history[-2:]
        depth_ratio = later[0] / max(earlier[0], 1e-8)
        seg_ratio = later[1] / max(earlier[1], 1e-8)
        delta = max(-60., min(60., (seg_ratio-depth_ratio)/self.temperature))
        depth = 2. / (1. + math.exp(delta))
        depth = min(2.-self.minimum, max(self.minimum, depth))
        return {'depth': depth, 'seg': 2.-depth}

    def record(self, depth, seg):
        if self.mode == 'fixed':
            return
        values = [float(depth), float(seg)]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('DWA monitors must be finite nonnegative losses')
        self.history.append(values)

    def state_dict(self):
        return {'mode': self.mode, 'temperature': self.temperature,
                'minimum': self.minimum, 'history': [v.copy() for v in self.history]}

    def load_state_dict(self, state):
        for name in ('mode', 'temperature', 'minimum'):
            if state[name] != getattr(self, name):
                raise ValueError(f'Task weighting {name} differs from checkpoint')
        self.history = []
        for depth, seg in state['history']:
            self.record(depth, seg)
