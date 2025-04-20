# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:18:48 2018

@author: 游侠-Speed
"""

from random import choice

class Random():
    
    def __init__(self, num_limit=50000):
        self.num_limit = num_limit
        self.x = [0]
        self.y = [0]
        
    def get_step(self):
        direction = choice([-1, 1])
        distence = choice([0, 1, 2, 3, 4, 5])
        return direction*distence
        
    def fill_walk(self):
        while len(self.x) < self.num_limit:
            x_step = self.get_step()
            y_step = self.get_step()
            x_next = self.x[-1] + x_step
            y_next = self.y[-1] + y_step
            self.x.append(x_next)
            self.y.append(y_next)