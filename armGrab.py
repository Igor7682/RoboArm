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

    def grab(angle1,angle2):
        
        side = "left"
        arm = Rooky2.Rooky('COM3', side)
        arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': 90
            },],2)
        arm.move_joints([{
                'name':'{0}_arm_2_joint'.format(side),
                'degree': angle1
            },],2)

        arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': angle2
            },],2)
        arm.move_joints([{
                'name':'{0}_arm_7_joint'.format(side),
                'degree': 74
            },],2)
        arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': 90
            },],2)
        arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': angle2
            },],2)
        arm.move_joints([{
                'name':'{0}_arm_7_joint'.format(side),
                'degree': 0
            },],2)
        time.sleep(2)

        #Груз зафиксирован, рука находится в стартовом положении

        arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': 90
            },],2)
        arm.move_joints([{
                'name':'{0}_arm_2_joint'.format(side),
                'degree': 0
            },],2)
        arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': 0
            },],2)
        
        #return True

        # Вернем все суставы в начальное положение
        arm.reset_joints()

        # Получим информацию о состоянии сервоприводов
        for i in arm.read_servos_data():
            print(i)


    def test():

        side = "left"
        arm = Rooky2.Rooky('COM3', side)
        arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': -20
            },],2)
        
        arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': 0
            },],2)
        
    def paint():

        side = "left"
        arm = Rooky2.Rooky('COM3', side)
        arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': 45
            },],2)
        
        arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(side),
                'degree': 0
            },],2)
    
if __name__ == "__main__":
    ang1 = 30
    ang2 = 63
    #paint()


