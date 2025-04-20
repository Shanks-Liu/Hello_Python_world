# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 10:55:47 2018

@author: 游侠-Speed
"""

def make_pizza(size, *toppings):
    print("Making a " + str(size) + "-inch pizza with the fllowing toppings:")
    for topping in toppings:
        print("\t-" + topping)