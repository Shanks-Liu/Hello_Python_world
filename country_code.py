# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 18:45:55 2018

@author: 游侠-Speed
"""
from pygal_maps_world.i18n import COUNTRIES

def get_country_code(country_name):
    for key, value in COUNTRIES.items():
        if country_name == "Yemen, Rep.":
            return "ye"
        if country_name == "China":
            return "cn"
        if country_name == value:
            return key
#        else:
#            return None  为什么加上else后全都是返回None
