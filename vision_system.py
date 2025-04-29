import cv2
import numpy as np
import os
from settings import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT
from newModel import predict

class VisionSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.detection_enabled = False
        self.current_frame = None
        self.detected_objects = []
        self.objInfo = []
        self.armPos = []
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            if self.detection_enabled:
                self.detect_objects()
            return True, frame
        return False, None
    
    def saveFrame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            filename = "sccreen.jpg"
            cv2.imwrite(filename,self.current_frame)
            return True
        return False


    def getObj(self):
        return self.objInfo
    
    def getPos(self):
        return self.armPos
    
    def predPos(self,x,y):
        pos = predict([x, y])
        return pos
    
    def detect_objects(self):
        """Обнаружение объектов"""
        if self.current_frame is None:
            return
            

        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        
        #green
        # lower_red = np.array([0, 120, 70])
        # upper_red = np.array([10, 255, 255])
        # mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        # lower_red = np.array([36,25,25])
        # upper_red = np.array([86, 255, 255])
        # mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        #blue
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([90, 50, 70])
        upper_red = np.array([128, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = cv2.bitwise_or(mask2, mask2)
        
        # Улучшение маски
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Нахождение контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.detected_objects = []
        self.objInfo.clear()
        self.armPos.clear()
        objNum  = 0
        for cnt in contours:
            objNum = objNum + 1
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

                self.objInfo.append((
                    objNum,
                    x,
                    y,
                    w,
                    h
                ))
                if x>0:
                    self.armPos.append(self.predPos(x,y))
                    print(self.armPos)


    
    
    def release(self):
        self.cap.release()