import cv2
import numpy as np
import os
from settings import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT
from newModel import predict
import time

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
            #self.current_frame = frame.copy()
            brightness = 10
            contrast = 2.3  
            # self.current_frame = cv2.addWeighted(frame, contrast, np.zeros(frame.shape, frame.dtype), 0, brightness)
            self.current_frame = cv2.detailEnhance(frame, sigma_s=150, sigma_r=0.15)
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
        lowerG = np.array([36,25,25])
        upperG = np.array([86, 255, 255])

        #blue1
        # lower = np.array([90, 50, 70])
        # upper = np.array([128, 255, 25])

        # lower = np.array([100, 150, 0])
        # upper = np.array([140, 255, 255])

        #blue2 
        lowerB = np.array([94, 80, 2])
        upperB= np.array([126, 255, 255])


        mask1 = cv2.inRange(hsv, lowerB, upperB)
        maskB = cv2.bitwise_or(mask1, mask1)

        mask2 = cv2.inRange(hsv, lowerG, upperG)
        maskG = cv2.bitwise_or(mask1, mask1)

        # Улучшение маски
        kernel = np.ones((5,5), np.uint8)
        maskB = cv2.morphologyEx(maskB, cv2.MORPH_OPEN, kernel)
        maskB = cv2.morphologyEx(maskB, cv2.MORPH_CLOSE, kernel)

        kernel = np.ones((5,5), np.uint8)
        maskG = cv2.morphologyEx(maskG, cv2.MORPH_OPEN, kernel)
        maskG = cv2.morphologyEx(maskG, cv2.MORPH_CLOSE, kernel)
        
        # Нахождение контуров
        contours1, _1 = cv2.findContours(maskG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _2 = cv2.findContours(maskB, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.detected_objects = []
        self.objInfo.clear()
        self.armPos.clear()
        objNum = 0

        

        for cnt in contours1:
            objNum = objNum + 1
            area = cv2.contourArea(cnt)
            if area > 500:  # Игнорируем маленькие объекты
                x, y, w, h = cv2.boundingRect(cnt)
                if x > 100:
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
                        'area': area,
                        'color': 'Green'
                    })

                    self.objInfo.append((
                        objNum,
                        x,
                        y,
                        w,
                        h,
                        'Green'
                    ))
                    if x > 0:
                        self.armPos.append(self.predPos(x,y))
                        #print(self.armPos)

        for cnt in contours2:
            objNum = objNum + 1
            area = cv2.contourArea(cnt)
            if area > 500:  # Игнорируем маленькие объекты
                x, y, w, h = cv2.boundingRect(cnt)
                if x > 100:
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
                        'area': area,
                        'color': 'Green'
                    })

                    self.objInfo.append((
                        objNum,
                        x,
                        y,
                        w,
                        h,
                        'Blue'
                    ))
                    if x > 0:
                        self.armPos.append(self.predPos(x,y))
                        #print(self.armPos)
                        


    
    
    def release(self):
        self.cap.release()