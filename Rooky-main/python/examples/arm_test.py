#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
"""Пример работы с манипулятором Rooky."""

__author__ = "Promobot"
__license__ = "Apache License, Version 2.0"
__status__ = "Production"
__url__ = "https://git.promo-bot.ru"
__version__ = "0.1.0"

# Импортируем необходимые библиотеки
# Библиотека для работы с Rooky
import Rooky

# Библиотека для работы с временными задержками
import time

# Создадим объект Rooky в соответствии с его типом: левая или правая
# '/dev/RS_485' - последовательный порт, для ubuntu по умолчанию - '/dev/RS_485'.
# 'right' - тип Rooky (left или right)
arm = Rooky.Rooky('/dev/RS_485','right')

while True:
	# Повернем сустав 1 на 30 градусов, максимальная скорость сервоприводов - 5 об/мин
	arm.move_joint('joint_1',5,30)

	# Получим данные от сервопривода
	print(arm.read_servos())

	# Задержка на 2 секунды
	time.sleep(2)

	# Вернем сустав в нулевое положение
	arm.move_joint('joint_1',5,0)

	# Задержка на 2 секунды
	time.sleep(2)
