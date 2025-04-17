# Параметры камеры
CAMERA_ID = 0  # ID камеры (0 для встроенной)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# FRAME_WIDTH = 1280
# FRAME_HEIGHT = 900

# Параметры манипулятора
ARM_CONFIG = {
    'upper_arm': 126.8,
    'shoulder': 256.1,
    'forearm': 149.1,
    'hand': 89,
    'fingers': 85,
    'joint_limits': {
        'shoulder_rotate': (-25, 134),
        'shoulder_move': (0, 83),
        'arm_rotate': (-89, 83),
        'elbow_move': (0, 80),
        'elbow_rotate': (-85, 85),
        'wrist_move': (-20, 31),
        'gripper': (0, 74)
    }
}


L1 = 126.8  # Надплечье (высота до плеча)
L2 = 256.1  # Плечо
L3 = 149.1  # Предплечье
L4 = 89.0   # Кисть



angle_limits = {
        'theta1': (-25, 134),  # Поворот основания
        'theta2': (0, 83),     # Подъем плеча
        'theta3': (-89, 83),   # Вращение плеча
        'theta4': (0, 80),     # Сгибание локтя
        'theta5': (-85, 85),   # Вращение локтя
        'theta6': (-20, 31)    # Наклон запястья
    }

# Цвета интерфейса
COLORS = {
    'background': '#2E2E2E',
    'text': '#FFFFFF',
    'button': '#3C3F41',
    'active': '#4E5254'
}