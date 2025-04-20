# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:32:41 2018

@author: 游侠-Speed
"""

class Restaurant():
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type
        self.number_served = 0
    def describe_restaurant(self):
        print(self.restaurant_name + " " + self.cuisine_type)
    def open_restaurant(self):
        print("The " + self.restaurant_name + " is opening")
    def read_number(self):
        print("The restaurant have " + 
              str(self.number_served) + 
              " people are eating.")
    def update_number(self,number):
        self.number_served += number
class IceCreamStand(Restaurant):
    def __init__(self,restaurant_name, cuisine_type, flavors):
        super().__init__(restaurant_name, cuisine_type)
        self.flavors = flavors
    def describe_icecream(self):
        for self.flavor in self.flavors:
            print(self.flavor)
flavors = ["orange", "strawberry"]