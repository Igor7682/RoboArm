#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

__author__ = "Promobot"
__license__ = "Apache License, Version 2.0"
__status__ = "Production"
__url__ = "https://git.promo-bot.ru"
__version__ = "0.1.0"

import Rooky2
import time
import random


# joint_limits['_arm_1_joint'] = [-25, 134]
# joint_limits['_arm_2_joint'] = [0, 83]
# joint_limits['_arm_3_joint'] = [-90, 83]
# joint_limits['_arm_4_joint'] = [0, 80]
# joint_limits['_arm_5_joint'] = [-86, 86]
# joint_limits['_arm_6_joint'] = [-20, 31]
# joint_limits['_arm_7_joint'] = [0, 74]

class arm():

    def __init__(self):
        self.side = "left"
        self.arm = Rooky2.Rooky('COM3', self.side)
        self.arm.set_touch_sensor_threshold(50)


    def armMove(self,angle1):
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 90
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_2_joint'.format(self.side),
                'degree': angle1
            },],2)
        # self.arm.move_joints([{
        #         'name':'{0}_arm_1_joint'.format(self.side),
        #         'degree': angle2
        #     },],2)
        # self.arm.move_joints([{
        #         'name':'{0}_arm_7_joint'.format(self.side),
        #         'degree': 74
        #     },],2)
        # self.arm.move_joints([{
        #         'name':'{0}_arm_1_joint'.format(self.side),
        #         'degree': 90
        #     },],2)


    def grab(self,angle1,angle2):

        self.armMove(angle1)
        self.grabObj(angle1,angle2)
        self.toTable()
        #self.placeObj(angle2)
        self.armReset()

    def armReset(self):
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 90
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_4_joint'.format(self.side),
                'degree': 0
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_5_joint'.format(self.side),
                'degree': 0
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_3_joint'.format(self.side),
                'degree': 0
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
        
        # self.arm.move_joints([{
        #         'name':'{0}_arm_2_joint'.format(self.side),
        #         'degree': angle1
        #     },],2)
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
        

    def to75(self):
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 90
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_2_joint'.format(self.side),
                'degree': 75
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_3_joint'.format(self.side),
                'degree': 80
            },],2)      
        self.arm.move_joints([{
                'name':'{0}_arm_5_joint'.format(self.side),
                'degree': -40
            },],2)   
        self.arm.move_joints([{
                'name':'{0}_arm_4_joint'.format(self.side),
                'degree': 40
            },],2)       
        
    def toTable(self):
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 90
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_2_joint'.format(self.side),
                'degree': 0
            },],2) 
        self.arm.move_joints([{
                'name':'{0}_arm_3_joint'.format(self.side),
                'degree': 20
            },],2) 
        self.arm.move_joints([{
                'name':'{0}_arm_4_joint'.format(self.side),
                'degree': 30
            },],2)  
        self.arm.move_joints([{
                'name':'{0}_arm_1_joint'.format(self.side),
                'degree': 80
            },],2)
        self.arm.move_joints([{
                'name':'{0}_arm_7_joint'.format(self.side),
                'degree': 0
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
        
    
if __name__ == "__main__":
    ang1 = 50
    ang2 = 57
    arm1 = arm()
    arm1.toTable()
    arm1.armReset()
    #arm.arm.arm.reset_joints()


