# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:07:27 2018

@author: 游侠-Speed
"""

import pygal
from die import Die

die1 = Die()
die2 = Die()

results = []
for num in range(1, 20):
    result = die1.roll()
    if result not in results:
        results.append(result)

print(results)
#    
#max_num = die1.num_size * die2.num_size 
#frequencies = []
#x_label = list(range(1, max_num+1))
#frequencies = [results.count(value) for value in list(range(1, max_num+1))]
#i = 0
#while i < len(x_label):
#    frequency = results.count(x_label[i])
#    if frequency == 0:
#        x_label.remove(x_label[i])
#        continue
#    else:
#        frequencies.append(frequency)
#        i += 1
#    
#hist = pygal.Bar()
#
#hist.title = "frequency graph"
#hist.x_title = "number"
#hist.y_title = "frequence"
#
#hist.x_labels = [ x for x in x_label] #横坐标列表元素用不用字符串都行
#hist.add("d1 * d2", frequencies)
#hist.render_to_file("frequncies.svg")