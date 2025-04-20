# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:29:47 2018

@author: 游侠-Speed
"""

import pygal
import requests

url = "https://api.github.com/search/repositories?q=language:python&sort=stars"
r = requests.get(url)
print(r.status_code)

response_dict = r.json()
print(len(response_dict))

repo_dicts = response_dict["item"]
print(len(repo_dicts))

names,attributes = [], []
attribute = {}
for repo_dict in repo_dicts:
    names.append(repo_dict["name"])
    attribute["value"] = repo_dict["stargazers_count"]
    attribute["label"] = repo_dict["description"]
    attribute["xlink"] = repo_dict["html_url"]
    attributes.append(attribute)
    
histogram = pygal.Bar()
histogram.x_labels = names
histogram.add(attributes)
histogram.render_to_file("python_hot.svg")