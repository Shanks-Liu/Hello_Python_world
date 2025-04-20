# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:42:15 2018

@author: 游侠-Speed
"""

class Employee():
    
    def __init__(self, first_name, last_name, annual_salary):
        self.first_name = first_name
        self.last_name = last_name
        self.annual_salary = annual_salary
        
    def give_raise(self, add=5000):
        self.annual_salary = self.annual_salary + add
        
    