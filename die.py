# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:36:16 2018

@author: 游侠-Speed
"""

from random import randint

class Die():
    
    def __init__(self, num_size=8):
        self.num_size = num_size
        
    def roll(self):
        return randint(1, self.num_size)

