# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 20:41:12 2019

@author: Shanks
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.matrics import accuracy_score


iris = load_iris()
data = iris.data

x = data[:, :-1]
y = data[:, -1]


