# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:33:52 2018

@author: 游侠-Speed
"""

import json
from pygal_maps_world.maps import World
from pygal.style import RotateStyle
from country_code import get_country_code

file_name = r"D:\Anaconda3\python_work\gdp_zip\data\gdp_json.json"
with open(file_name) as f:
    gdpls = json.load(f)

gdp_values = {}    
for gdpd in gdpls:
    if gdpd["Year"] == 2016:
        code = get_country_code(gdpd["Country Name"])
        gdp_values[code] = int(float(gdpd["Value"]))

color = RotateStyle("#336699")        
gdp_map = World(style=color)
gdp_map.add("2016", gdp_values)
gdp_map.render_to_file("gdp_map.svg")