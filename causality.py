# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:55:02 2018

@author: 游侠-Speed
"""

import numpy as np
import pandas as pd

from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest

data = pd.read_excel("fujianshengyinguo.xlsx")
#data = pd.read_excel("svrexample.xlsx")

data.columns = ["x1", "x2", "x3", 'x4', 'x5']



# generate some toy data:
#SIZE = 2000
#x1 = numpy.random.normal(size=SIZE)
#x2 = x1 + numpy.random.normal(size=SIZE)
#x3 = x1 + numpy.random.normal(size=SIZE)
#x4 = x2 + x3 + numpy.random.normal(size=SIZE)
#x5 = x4 + numpy.random.normal(size=SIZE)

# load the data into a dataframe:
#X = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3, 'x4' : x4, 'x5' : x5})

# define the variable types: 'c' is 'continuous'.  The variables defined here
# are the ones the search is performed over  -- NOT all the variables defined
# in the data frame.
variable_types = {'x1' : 'c', 'x2' : 'c', 'x3' : 'c', 'x4' : 'c', 'x5' : 'c'}

# run the search
ic_algorithm = IC(RobustRegressionTest)
graph = ic_algorithm.search(data, variable_types)

print(graph.edges(data=True))