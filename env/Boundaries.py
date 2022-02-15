import pybullet as p

class obstacles:
    def __init__(self,bc):
        # Here is to load the square, but i found round is better to calculate
        self.bc = bc
        self.bc.loadURDF("env/obstacle/circle_big.urdf",useFixedBase=1)
        self.bc.loadURDF("env/obstacle/circle_small.urdf",useFixedBase=1)
