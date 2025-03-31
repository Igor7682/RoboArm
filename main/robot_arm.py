import numpy as np
from scipy.optimize import minimize
from settings import ARM_CONFIG

class RobotArm:
    def __init__(self):
        self.config = ARM_CONFIG
        self.current_angles = [0, 0, 0, 0, 0, 0]
        
    def forward_kinematics(self, angles):
        """Прямая кинематика (упрощенная реализация)"""
        angles_rad = np.radians(angles)
        x = sum([
            self.config['upper_arm'] * np.cos(angles_rad[0]) * np.cos(angles_rad[1]),
            self.config['shoulder'] * np.cos(angles_rad[0]) * np.cos(angles_rad[1] + angles_rad[3]),
            self.config['forearm'] * np.cos(angles_rad[0]) * np.cos(angles_rad[1] + angles_rad[3] + angles_rad[5])
        ])
        
        y = sum([
            self.config['upper_arm'] * np.sin(angles_rad[0]) * np.cos(angles_rad[1]),
            self.config['shoulder'] * np.sin(angles_rad[0]) * np.cos(angles_rad[1] + angles_rad[3]),
            self.config['forearm'] * np.sin(angles_rad[0]) * np.cos(angles_rad[1] + angles_rad[3] + angles_rad[5])
        ])
        
        z = sum([
            self.config['upper_arm'] * np.sin(angles_rad[1]),
            self.config['shoulder'] * np.sin(angles_rad[1] + angles_rad[3]),
            self.config['forearm'] * np.sin(angles_rad[1] + angles_rad[3] + angles_rad[5]),
            self.config['hand']
        ])
        
        return np.array([x, y, z])
    
    def inverse_kinematics(self, target_pos):
        """Обратная кинематика"""
        bounds = [
            self.config['joint_limits']['shoulder_rotate'],
            self.config['joint_limits']['shoulder_move'],
            self.config['joint_limits']['arm_rotate'],
            self.config['joint_limits']['elbow_move'],
            self.config['joint_limits']['elbow_rotate'],
            self.config['joint_limits']['wrist_move']
        ]
        
        def objective(angles):
            pos = self.forward_kinematics(angles)
            return np.linalg.norm(pos - target_pos)
            
        result = minimize(objective, self.current_angles, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            return np.degrees(result.x)
        raise ValueError("IK solution not found")
    
    def move_to(self, x, y, z):
        """Команда перемещения"""
        target = np.array([x, y, z])
        try:
            angles = self.inverse_kinematics(target)
            self.current_angles = angles
            print(f"Moving to: {angles}")  # В реальности отправка команд на манипулятор
            return True
        except Exception as e:
            print(f"Move error: {str(e)}")
            return False
    
    def grab(self, force=50):
        """Команда захвата"""
        angle = self.config['joint_limits']['gripper'][1] * (force / 100)
        print(f"Grabbing with angle: {angle}")
    
    def release(self):
        """Команда освобождения"""
        print("Releasing object")