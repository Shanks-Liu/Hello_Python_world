# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 09:46:56 2018

@author: 游侠-Speed
"""


def get_city(city, country, population=""):
    if population:
        location = city + " " + country + " population: " + str(population)
        return location
    else:
        location = city + " " + country
        return location
