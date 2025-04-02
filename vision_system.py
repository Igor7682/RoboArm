import cv2
import numpy as np
from settings import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT

class VisionSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.detection_enabled = False
        self.current_frame = None
        self.detected_objects = []
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            if self.detection_enabled:
                self.detect_objects()
            return True, frame
        return False, None
    

    
    def detect_objects(self):
        """Обнаружение объектов"""
        if self.current_frame is None:
            return
            
        # Преобразование в HSV для лучшего выделения объектов
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        
        # Создание маски (для красных объектов)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Улучшение маски
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Нахождение контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.detected_objects = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Игнорируем маленькие объекты
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Вычисление центра масс
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                self.detected_objects.append({
                    'position': (cx, cy),
                    'size': (w, h),
                    'contour': cnt,
                    'area': area
                })
    
    def release(self):
        self.cap.release()