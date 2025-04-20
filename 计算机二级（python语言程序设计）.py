# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:49:12 2019

@author: Shanks
"""

#斐波那契数列
a, b = 0, 1
while a < 1000:
    print(a, end=",")
    a, b = b, a+b
    
#绘制五角红星
from turtle import *
color('red', 'red')
begin_fill()
for i in range(5):
    fd(200)
    rt(144)
end_fill()
done()

#程序运行时间
import time
limit = 10 * 1000 * 1000
start = time.perf_counter()
while True:
    limit -= 1
    if limit <= 0:
        break
delta = time.perf_counter() - start
print('程序运行的时间是：{}秒'.format(delta))

