# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:46:48 2018

@author: 游侠-Speed
"""

import unittest
from employee import Employee

class TestEmployee(unittest.TestCase):
    
    def setUp(self):
        self.my_employee = Employee("shanks", "liu", 500000)
        
    def test_give_default_raise(self):
        self.my_employee.give_raise()
        self.assertEqual(505000, self.my_employee.annual_salary)
        
    def test_give_custom_raise(self):
        self.my_employee.give_raise(50000)
        self.assertEqual(550000, self.my_employee.annual_salary)

unittest.main()