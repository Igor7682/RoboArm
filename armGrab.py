#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

__author__ = "Promobot"
__license__ = "Apache License, Version 2.0"
__status__ = "Production"
__url__ = "https://git.promo-bot.ru"
__version__ = "0.1.0"

# Импортируем необходимые библиотеки
# Библиотека для работы с Rooky версии 2
import Rooky2

# Библиотека для работы с временными задержками
import time
import random

# Укажем тип Rooky left или right


#83-75-67-59
#37-44-51-60
# Создадим объект Rooky в соответствии с его типом: левая или правая
# '/dev/RS_485' - последовательный порт, для ubuntu по умолчанию - '/dev/RS_485'.





#1 - рука вверх.вниз 160
#2 - рука влево.вправо 83
#3 - вращение руки 173
#4 - локоть влево.вправо 80
#5 - вращение локтя 172
#6 - кисть вверх.вниз 51
#7 - пальцы 74

class arm():

    def __init__(self):
        self.side = "left"
        self.arm = Rooky2.Rooky('COM3', self.side)
        self.arm.set_touch_sensor_threshold(50)


    def armMove(self,angle1,angle2):
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 90
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_2_joint'.format(self.side),
                'degree': angle1
            },],2)

        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': angle2
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_7_joint'.format(self.side),
                'degree': 74
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 90
            },],2)


    def grab(self,angle1,angle2):

        self.armMove(angle1,angle2)
        # while True:
        #     if self.arm.is_touched():
        #         self.placeObj(angle2)
        #         break
        #     else:
        #         angle1 = angle1 + random.uniform(-0.5, 0.5)
        #         angle2 = angle2 + random.uniform(-0.5, 0.5)
        #         self.arm.move_joints([{
        #         'name':'{0}_arm_7_joint'.format(self.side),
        #         'degree': 0
        #     },],2)
        #         self.grabObj(angle1,angle2)


        self.grabObj()
        self.placeObj()

                

        #time.sleep(2)

        self.armReset()

        #self.arm.reset_joints()

        # Получим информацию о состоянии сервоприводов
        # for i in arm.read_servos_data():
        #     print(i)


    def armReset(self):
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 90
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_2_joint'.format(self.side),
                'degree': 0
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 0
            },],2)
        #self.arm.reset_joints()

    def grabObj(self,angle1,angle2):
        
        self.arm.move_joints([{
                'name':'{0}_arm_2_joint'.format(self.side),
                'degree': angle1
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': angle2
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_7_joint'.format(self.side),
                'degree': 74
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 90
            },],2)
        

    def placeObj(self,angle2):
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': angle2
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_7_joint'.format(self.side),
                'degree': 0
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 90
            },],2)
        

    def test(self):

        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': -20
            },],2)
        
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 0
            },],2)
        
    def paint(self):

        side = "left"
        self.arm = Rooky2.Rooky('COM3', side)
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': 45
            },],2)
        
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': 0
            },],2)
    
if __name__ == "__main__":
    # ang1 = 30
    # ang2 = 63
    arm = arm()
    #arm.arm.arm.reset_joints()


