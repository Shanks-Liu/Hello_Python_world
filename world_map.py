# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 17:37:33 2018

@author: 游侠-Speed
"""

#import pygal
import json
from country_code import get_country_code
from pygal.style import RotateStyle
from pygal.style import LightColorizedStyle
from pygal_maps_world import maps

population = {}

with open("population-figures-by-country-csv_json.json") as f:
    population_datas = json.load(f)    
for population_dic in population_datas[:]:
    if population_dic["Population_in_2016"]:
        code = get_country_code(population_dic["Country_Name"])
        print(code)
        if code:
            population[code] = int(float(population_dic["Population_in_2016"]))

color = RotateStyle("#113388",base_style=LightColorizedStyle)
m_map = maps.World(style=color)
m_map.title = "World Population Graph"
m_map.add("2015", population)
m_map.render_to_file("population.svg")