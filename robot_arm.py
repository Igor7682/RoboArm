import numpy as np
from scipy.optimize import minimize
import settings
from math import cos, sin, atan2, acos, sqrt, degrees, radians


class RobotArm:
    def __init__(self):
        #self.config = ARM_CONFIG
        #self.vision= vision_System
        self.current_angles = [0, 0, 0, 0, 0, 0]


        
