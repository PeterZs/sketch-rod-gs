import numpy as np
from del_fem_numpy.Rod3Darboux import Simulator


class PolyLine:
    polyline: np.ndarray
    rest_polyline: np.ndarray
    stationary_steps: int
    prev_polyline: np.ndarray

    def __init__(self):
        self.reset_polyline()

    def reset_polyline(self):
        self.polyline = np.array([])
        self.rest_polyline = self.polyline.copy()
        self.stationary_steps = 0
        self.prev_polyline = None

    def set_polyline(self, polyline: np.ndarray):
        self.polyline = polyline
        self.rest_polyline = self.polyline.copy()
        self.stationary_steps = 0
        self.prev_polyline = None

        self.simulator = Simulator(self.polyline)
        self.simulator.vtx2isfix[0][:] = 1
        self.simulator.vtx2isfix[1][0:3] = 1
        self.simulator.vtx2isfix[-1][:] = 1
        self.simulator.vtx2isfix[-2][0:3] = 1

    @property
    def get_polyline(self) -> np.ndarray:
        return self.polyline

    @property
    def get_rest_polyline(self) -> np.ndarray:
        return self.rest_polyline

    def is_polyline_exist(self) -> bool:
        return 0 < self.polyline.shape[0]

    def simulate(self, time_step: float, pull_point_id: int | None, pos_goal: np.ndarray | None):
        # Update polyline
        if pull_point_id is not None:
            assert len(pos_goal.shape) == 1 and pos_goal.shape[0] == 3
            self.simulator.solve_dynamic(time_step, (pull_point_id, pos_goal))
        else:
            self.simulator.solve_dynamic(time_step, None)
        self.prev_polyline = self.polyline
        self.polyline = self.simulator.vtx2xyz_def

    def is_anim_finished(self, diff_threshhold=1e-2, required_step=10):
        diff_rest = np.sum((self.polyline - self.rest_polyline) ** 2)
        if diff_rest < diff_threshhold:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0

        if self.stationary_steps >= required_step:
            self.stationary_steps = 0
            return True

        return False
